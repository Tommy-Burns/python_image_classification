from sklearn import svm
from skimage import exposure
from skimage.segmentation import quickshift, slic
import time
import scipy
import os
import numpy as np
from osgeo import gdal, ogr
from sklearn import metrics

composite_img = "./images/clipped.tif"

driverTiff = gdal.GetDriverByName('GTiff')
composite_ds = gdal.Open(composite_img)
nbands = composite_ds.RasterCount
band_data = []
print('bands', composite_ds.RasterCount, 'rows', composite_ds.RasterYSize, 'columns',
      composite_ds.RasterXSize)
for i in range(1, nbands + 1):
    band = composite_ds.GetRasterBand(i).ReadAsArray()
    band_data.append(band)
band_data = np.dstack(band_data)

# scale image values from 0.0 - 1.0
img = exposure.rescale_intensity(band_data)

# do segmentation multiple options with slic
seg_start = time.time()
segments = slic(img, n_segments=500000, compactness=0.1)
print('segments complete', time.time() - seg_start)


def segment_features(segment_pixels):
    features = []
    npixels, nbands = segment_pixels.shape
    for b in range(nbands):
        stats = scipy.stats.describe(segment_pixels[:, b])
        band_stats = list(stats.minmax) + list(stats)[2:]
        if npixels == 1:
            # in this case the variance = nan, change it 0.0
            band_stats[3] = 0.0
        features += band_stats
    return features


start = time.time()
segment_ids = np.unique(segments)
objects = []
object_ids = []
for id in segment_ids:
    segment_pixels = img[segments == id]
    object_features = segment_features(segment_pixels)
    objects.append(object_features)
    object_ids.append(id)
finish = time.time()
minute = (finish - start) / 60
seconds = (minute - int(minute)) * 60
print(f"created {len(objects)} objects with {len(objects[0])} variables in {minute} minutes and {seconds} seconds.")

# save segments to raster
os.mkdir('images') if not os.path.isdir('./images') else None
segments_fn = os.path.join(os.getcwd(), os.path.join('images', 'segments.tif'))
segments_ds = driverTiff.Create(segments_fn, composite_ds.RasterXSize, composite_ds.RasterYSize,
                                1, gdal.GDT_Float32)
segments_ds.SetGeoTransform(composite_ds.GetGeoTransform())
segments_ds.SetProjection(composite_ds.GetProjectionRef())
segments_ds.GetRasterBand(1).WriteArray(segments)
segments_ds = None

# RASTERIZE TRUTH DATA

# open clipped image as a gdal raster dataset
clip_in = composite_img
clip_ds = gdal.Open(clip_in)

# open the points file to use for training data
train_fn = r"./test_train/training.shp"
train_ds = ogr.Open(train_fn)
lyr = train_ds.GetLayer()
# create a new raster layer in memory
driver = gdal.GetDriverByName('MEM')
target_ds = driver.Create('', clip_ds.RasterXSize, clip_ds.RasterYSize, 1, gdal.GDT_UInt16)
target_ds.SetGeoTransform(clip_ds.GetGeoTransform())
target_ds.SetProjection(clip_ds.GetProjection())
# rasterize the training points
options = ['ATTRIBUTE=id']
gdal.RasterizeLayer(target_ds, [1], lyr, options=options)

"""
Get segments representing each land cover classification type and ensure no segment represents more than one class.
"""

ground_truth = target_ds.GetRasterBand(1).ReadAsArray()

classes = np.unique(ground_truth)[1:]
print('class values', classes)

segments_per_class = {}

for klass in classes:
    segments_of_class = segments[ground_truth == klass]
    segments_per_class[klass] = set(segments_of_class)
    print("Training segments for class", klass, ":", len(segments_of_class))

intersection = set()
accum = set()

for class_segments in segments_per_class.values():
    intersection |= accum.intersection(class_segments)
    accum |= class_segments
assert len(intersection) == 0, "Segment(s) represent multiple classes"

# Now to the Main deal! The Support vector classification
train_img = np.copy(segments)
threshold = train_img.max() + 1

for klass in classes:
    class_label = threshold + klass
    for segment_id in segments_per_class[klass]:
        train_img[train_img == segment_id] = class_label

train_img[train_img <= threshold] = 0
train_img[train_img > threshold] -= threshold

training_objects = []
training_labels = []

for klass in classes:
    class_train_object = [v for i, v in enumerate(objects) if segment_ids[i] in segments_per_class[klass]]
    training_labels += [klass] * len(class_train_object)
    training_objects += class_train_object
    print('Training objects for class', klass, ':', len(class_train_object))

print('Staring the Support Vector Machine')
classifier = svm.SVC(kernel='rbf', gamma='scale')
print('Fitting the Support Vector Machine')
classifier.fit(np.ndarray(training_objects).reshape(-1, 1), training_labels)
print('Predicting Classifications')
predicted = classifier.predict(objects)
print('Predictions done...')
print('\nApplying predictions to numpy array. This will take some time....')
clf = np.copy(segments)
for segment_id, klass in zip(segment_ids, predicted):
    clf[clf == segment_id] = klass

print('Predictions applied to numpy array')

mask = np.sum(img, axis=2)
mask[mask > 0.0] = 1.0
mask[mask == 0.0] = -1.0
clf = np.multiply(clf, mask)
clf[clf < 0] = -9999.0

print('Saving classification to raster with gdal')
os.mkdir('./classified') if not os.path.isdir('./classified') else None
clfds = driverTiff.Create(r"./classified/classified.tif",
                          composite_ds.RasterXSize, composite_ds.RasterYSize,
                          1, gdal.GDT_Float32)
clfds.SetGeoTransform(composite_ds.GetGeoTransform())
clfds.SetProjection(composite_ds.GetProjection())
clfds.GetRasterBand(1).SetNoDataValue(-9999.0)
clfds.GetRasterBand(1).WriteArray(clf)
clfds = None

print("Classification complete...")

# Computing Accuracies

naip_fn = r"path to clipped composite image (must be a .tif file)"

driverTiff = gdal.GetDriverByName('GTiff')
naip_ds = gdal.Open(naip_fn)

test_fn = r"./images/testing.shp"
test_ds = ogr.Open(test_fn)
lyr = test_ds.GetLayer()
driver = gdal.GetDriverByName('MEM')
target_ds = driver.Create('', naip_ds.RasterXSize, naip_ds.RasterYSize, 1, gdal.GDT_UInt16)
target_ds.SetGeoTransform(naip_ds.GetGeoTransform())
target_ds.SetProjection(naip_ds.GetProjection())
options = ['ATTRIBUTE=id']
gdal.RasterizeLayer(target_ds, [1], lyr, options=options)

truth = target_ds.GetRasterBand(1).ReadAsArray()

pred_ds = gdal.Open(r"path to classified image")
pred = pred_ds.GetRasterBand(1).ReadAsArray()

idx = np.nonzero(truth)

cm = metrics.confusion_matrix(truth[idx], pred[idx])
cls_accuracy = metrics.accuracy_score(truth[idx], pred[idx], normalize=True)
kappa = metrics.cohen_kappa_score(truth[idx], pred[idx])

# pixel accuracy
print(f"*******Confusion Matrix********\n{cm}\n********************************")

# print(cm.diagonal())
# print(cm.sum(axis=0))

accuracy = cm.diagonal() / cm.sum(axis=0)
# print(accuracy)
for i in accuracy:
    print(f"Accuracy for class{i}: {round(i * 100, 3)} %")

print(f"\nClassification Accuracy: {round(cls_accuracy * 100, 3)} %")
print(f"Kappa: {kappa}")
