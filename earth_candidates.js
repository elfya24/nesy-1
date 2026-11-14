// =====================================================================
// earth_candidates — 5 bölge, 5 km x 5 km grid, 13 sütunlu CSV (Drive)
// =====================================================================

// ------------------------- AYARLAR -----------------------------------
var START = '2023-01-01';
var END   = '2025-01-01';      // END dahil değil
var CELL_M = 5000;             // hücre boyutu (m)
var SCALE = 90;                // hücre içi örnekleme çözünürlüğü (m)
var NDVI_MAX = 0.15;           // bunun ÜSTÜ bitkili sayılıp maskelenir
var MIN_VALID_FRAC = 0.5;      // hücrede maske sonrası kalması gereken min. piksel oranı

// bbox = [batı, güney, doğu, kuzey]
var REGIONS = [
  {name: 'Ic_Dogu_Anadolu_Volkanik', bbox: [33.0, 37.5, 44.5, 40.0],     epsg: 'EPSG:32637'},
  {name: 'Harrat_Khaybar',           bbox: [39.0, 24.5, 41.0, 26.5],     epsg: 'EPSG:32637'},
  {name: 'Beishan_Gobi',             bbox: [95.0, 40.5, 99.0, 42.5],     epsg: 'EPSG:32647'},
  {name: 'Tibet_Qaidam',             bbox: [91.0, 36.5, 95.0, 38.5],     epsg: 'EPSG:32646'},
  {name: 'And_Altiplano',            bbox: [-68.5, -21.0, -66.5, -18.5], epsg: 'EPSG:32719'}
];

var COLUMNS = ['lat', 'lon', 'elevation_m', 'slope_deg', 'tri_m', 'delta_t_k',
               'mafic_index', 'clay_hydrated_index', 'subsurface_ice_moisture',
               'ndvi', 'albedo', 'thermal_inertia_proxy', 'landcover_class'];

// ------------------------- BÖLGELER / GRID ----------------------------
var rects = REGIONS.map(function (r) {
  return ee.Feature(ee.Geometry.Rectangle(r.bbox, 'EPSG:4326', false), {name: r.name});
});
var regionsFC = ee.FeatureCollection(rects);
var allBounds = regionsFC.geometry();

function makeGrid(r) {
  var rect = ee.Geometry.Rectangle(r.bbox, 'EPSG:4326', false);
  var proj = ee.Projection(r.epsg).atScale(CELL_M);
  var grid = rect.coveringGrid(proj);
  var west = r.bbox[0], south = r.bbox[1], east = r.bbox[2], north = r.bbox[3];

  grid = grid.map(function (c) {
    var poly = c.geometry().transform('EPSG:4326', 1);
    var ctr = poly.centroid(1).coordinates();
    return ee.Feature(poly, {
      region: r.name,
      lon: ctr.get(0),
      lat: ctr.get(1)
    });
  });
  // Sadece merkezi dikdörtgenin içinde kalan hücreleri tut
  return grid.filter(ee.Filter.and(
    ee.Filter.gte('lon', west), ee.Filter.lt('lon', east),
    ee.Filter.gte('lat', south), ee.Filter.lt('lat', north)
  ));
}

var grids = REGIONS.map(makeGrid);
var cells = ee.FeatureCollection(grids).flatten();

// ------------------------- VERİ KATMANLARI ----------------------------
// ESA WorldCover (su = 80, şehir = 50)
var wc = ee.ImageCollection('ESA/WorldCover/v200').first().select('Map');
var notWaterUrban = wc.neq(80).and(wc.neq(50));

// Landsat 9 L2 — bulut/gölge maskesi + ölçek faktörleri
function prepL9(img) {
  var qa = img.select('QA_PIXEL');
  var clear = qa.bitwiseAnd(1 << 0).eq(0)    // fill
    .and(qa.bitwiseAnd(1 << 1).eq(0))        // dilated cloud
    .and(qa.bitwiseAnd(1 << 2).eq(0))        // cirrus
    .and(qa.bitwiseAnd(1 << 3).eq(0))        // cloud
    .and(qa.bitwiseAnd(1 << 4).eq(0));       // cloud shadow
  var optical = img.select('SR_B.').multiply(0.0000275).add(-0.2);
  return img.addBands(optical, null, true).updateMask(clear);
}

var l9 = ee.ImageCollection('LANDSAT/LC09/C02/T1_L2')
  .filterDate(START, END)
  .filterBounds(allBounds)
  .map(prepL9)
  .select(['SR_B2', 'SR_B3', 'SR_B4', 'SR_B5', 'SR_B6', 'SR_B7'])
  .median();

var blue = l9.select('SR_B2'), green = l9.select('SR_B3'), red = l9.select('SR_B4');
var nir = l9.select('SR_B5'), swir1 = l9.select('SR_B6'), swir2 = l9.select('SR_B7');

var ndvi = nir.subtract(red).divide(nir.add(red)).rename('ndvi');
var mafic = red.divide(blue.updateMask(blue.gt(0))).rename('mafic_index');                 // SR_B4 / SR_B2
var clay = swir1.divide(swir2.updateMask(swir2.gt(0))).rename('clay_hydrated_index');      // SR_B6 / SR_B7
var albedo = blue.add(green).add(red).divide(3).rename('albedo');                           // görünür bant ortalaması

// NASADEM
var dem = ee.Image('NASA/NASADEM_HGT/001').select('elevation');
var slope = ee.Terrain.slope(dem);

// MODIS LST (ham DN * 0.02 = Kelvin)
var modis = ee.ImageCollection('MODIS/061/MOD11A1')
  .filterDate(START, END)
  .select(['LST_Day_1km', 'LST_Night_1km']);
var dayMean = modis.select('LST_Day_1km').mean();
var nightMean = modis.select('LST_Night_1km').mean();
var deltaT = dayMean.subtract(nightMean).multiply(0.02).rename('delta_t_k');
var tNight = nightMean.multiply(0.02).rename('t_night');

// SMAP L4 yüzey nemi
var smap = ee.ImageCollection('NASA/SMAP/SPL4SMGP/007')
  .filterDate(START, END)
  .select('sm_surface')
  .mean()
  .rename('subsurface_ice_moisture');

// ------------------------- PİKSEL MASKESİ -----------------------------
// Kalan piksel: su/şehir değil VE NDVI <= eşik
var valid = ndvi.lte(NDVI_MAX).and(notWaterUrban);
var validFrac = valid.unmask(0, false).rename('valid_frac');

// Bant sırası reducer forEach sırasıyla birebir eşleşmek zorunda
var meanNames = ['elevation_m', 'slope_deg', 'delta_t_k', 'mafic_index',
                 'clay_hydrated_index', 'subsurface_ice_moisture', 'ndvi',
                 'albedo', 't_night'];

var meanBands = ee.Image.cat([
  dem.rename('elevation_m'),
  slope.rename('slope_deg'),
  deltaT,
  mafic,
  clay,
  smap,
  ndvi,
  albedo,
  tNight
]).select(meanNames).updateMask(valid);

var stack = ee.Image.cat([
  meanBands,                                    // 9 bant (ortalama)
  validFrac,                                    // maskelenmemiş, oran için
  dem.rename('tri_m').updateMask(valid),        // yükseklik std sapması için
  wc.rename('landcover_class').updateMask(valid)
]);

// ÇÖZÜM: tüm reducer'lar .unweighted() — ağırlıklı/ağırlıksız karışımı hatası biter.
var meanReducer = ee.Reducer.mean().unweighted()
  .forEach(meanNames.concat(['valid_frac']));
var stdReducer  = ee.Reducer.stdDev().unweighted()
  .forEach(['tri_m']);
var modeReducer = ee.Reducer.mode().unweighted()
  .forEach(['landcover_class']);

var reducer = meanReducer
  .combine(stdReducer, '', false)
  .combine(modeReducer, '', false);

// ------------------------- HÜCRE İSTATİSTİKLERİ ----------------------
var stats = stack.reduceRegions({
  collection: cells,
  reducer: reducer,
  scale: SCALE,
  tileScale: 4
});

var rawCols = meanNames.concat(['valid_frac', 'tri_m', 'landcover_class']);

var result = stats
  .filter(ee.Filter.notNull(rawCols))
  .filter(ee.Filter.gte('valid_frac', MIN_VALID_FRAC))
  .map(function (f) {
    var tn = ee.Number(f.get('t_night'));
    var dt = ee.Number(f.get('delta_t_k'));
    var ti = ee.Algorithms.If(dt.gt(0.1), tn.divide(dt), null);   // T_night / delta_t_k
    return f.set('thermal_inertia_proxy', ti);
  })
  .filter(ee.Filter.notNull(['thermal_inertia_proxy']))
  .select(COLUMNS, null, false);   // sadece 13 sütun, geometri yok

// ------------------------- KONTROL / HARİTA ---------------------------
Map.centerObject(allBounds, 3);
Map.addLayer(regionsFC.style({color: 'red', fillColor: '00000000', width: 2}), {}, 'Bölgeler');
Map.addLayer(valid.selfMask(), {palette: ['00ff00'], opacity: 0.4}, 'Kalan (geçerli) pikseller', false);

print('Toplam grid hücresi:', cells.size());
print('İlk hücre alanı (m²) ~25e6 olmalı:', cells.first().geometry().area(1));

// ------------------------- EXPORT -------------------------------------
Export.table.toDrive({
  collection: result,
  description: 'earth_candidates',
  fileNamePrefix: 'earth_candidates',
  fileFormat: 'CSV',
  selectors: COLUMNS
});
