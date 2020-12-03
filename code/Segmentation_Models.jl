using random
using json
using h5py
using Images
using ImageSegmentation
using FileIO
using ImageView
using glob

using Colors
using PatchLibrary
shr = unseeded_region_growing(hr,0.15) #threshold for merging pixel to clusters
map(i->segment_mean(shr,i), labels_map(shr))
slh = unseeded_region_growing(lh,0.15)


    train_data = glob('train_data/**')
    patches = PatchLibrary((33,33),slh, train_data, 50000)
    X,y = patches.make_training_patches()

    model = ImageSegmentation()
    model.fit_model(X, y)
    model.save_model('models/example')

    # tests = glob('test_data/2_*')
    # test_sort = sorted(tests, key= lambda x: int(x[12:-4]))
    # model = BasicModel(loaded_model=True)
    # segmented_images = []
    # for slice in test_sort[15:145]:
    #     segmented_images.append(model.show_segmented_image(slice))
