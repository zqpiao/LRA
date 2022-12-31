from .faster_rcnn import FasterRCNN
# from .faster_rcnn_softmax import FasterRCNN

def build_detectors(cfg):
    model = FasterRCNN(cfg)
    return model
