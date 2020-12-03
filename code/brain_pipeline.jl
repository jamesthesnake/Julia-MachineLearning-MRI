using communicate
using Random
using glob
seed=Random.seed(5) # for reproducibility
Struct BrianPipeline
    int images;
    labels = glob('Original_Data/Training/HGG/**/*more*/**.mha')
    save_labels(labels)
#add s3 upload here
