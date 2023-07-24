from collections.abc import Iterable

from detection_helpers import *
from tracking_helpers import *
from bridge_wrapper import *

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--yolo_model', type=str, default='./weights/yolov7x.pt', help='yolo model.pt path(s)')
    parser.add_argument('--reid_model', type=str, default='./deep_sort/model_weights/mars-small128.pb', help='reID model.pb path(s)')
    parser.add_argument('--classes', nargs='+', type=int, default=[0], help='Filter by class from COCO. can be in the format [0] or [0,1,2] etc')
    parser.add_argument('--video', type=str, help='path to input video or set to 0 for webcam', required=True)
    parser.add_argument('--output', type=str, help='path to output video', required=True)
    parser.add_argument('--skip_frames', type=int, default=0, help="Skip every nth frame. After saving the video, it'll have very visuals experience due to skipped frames")
    parser.add_argument('--show_live', type=bool, default=False, help="Whether to show live video tracking. Press the key 'q' to quit")
    parser.add_argument('--count_objects', type=bool, default=True, help='count objects being tracked on screen')
    parser.add_argument('--verbose', type=int, default=2, help='print details on the screen allowed values 0,1,2')

    args = parser.parse_args()

    detector = Detector(classes=args.classes)
    detector.load_model(args.yolo_model) # pass the path to the trained weight file

    # Initialise  class that binds detector and tracker in one class
    tracker = YOLOv7_DeepSORT(reID_model_path=args.reid_model, detector=detector)

    f = open(f"{args.output}_mot.txt", 'w'); f.close()

    # output = None will not save the output video
    tracker.track_video(video=args.video,
                        output=args.output,
                        show_live=args.show_live,
                        skip_frames=args.skip_frames,
                        count_objects=args.count_objects,
                        verbose=args.verbose)