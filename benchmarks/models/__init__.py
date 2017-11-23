def get_model(model_name):
    if model_name == "alexnet":
        from . import alex
        model = alex.Alex()
    elif model_name == "googlenet":
        from . import googlenet
        model = googlenet.GoogLeNet()
    elif model_name == "vgga":
        from . import vgga
        model = vgga.vgga()
    elif model_name == "overfeat":
        from . import overfeat
        model = overfeat.overfeat()
    elif model_name == "resnet50":
        from . import resnet50
        model = resnet50.ResNet()
    elif model_name == "vgg16":
        from . import vgg16
        model = vgg16.VGG16()
    else:
        raise ValueError('Invalid architecture name')
    return model
