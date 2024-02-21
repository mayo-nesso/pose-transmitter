import argparse
import threading

from inference_hub import hub
from qudp_transmitter import QUDPTransmitter
from video_pose import VideoPose


def parse_arguments():
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(description="Description of your program")
    parser.add_argument("--debug", action="store_true", help="Enable debug mode")
    parser.add_argument("--source", default=0, help="Source file for video input, or 0 for webcam")
    parser.add_argument("--host_ip", default="127.0.0.1", help="Host IP address")
    parser.add_argument("--host_port", type=int, default=4900, help="Host port number")
    parser.add_argument(
        "--infer_model",
        type=str,
        default="lightning",
        help="Movenet Inference Model to use; Options are: \
                            lightning, lightning_lite8, lightning_lite16, \
                            thunder, thunder_lite8, thunder_lite16",
    )
    return parser.parse_args()


def retrive_hub(infer_model):
    """Retrieve the inference hub based on the specified model."""
    match infer_model:
        case "lightning":
            return hub.init_lightning()
        case "lightning_lite8":
            return hub.init_lightning_lite8()
        case "lightning_lite16":
            return hub.init_lightning_lite16()
        case "thunder":
            return hub.init_thunder()
        case "thunder_lite8":
            return hub.init_thunder_lite8()
        case "thunder_lite16":
            return hub.init_thunder_lite16()
        case _:
            raise Exception(
                "Invalid inference model. Options are: \n \
                            \t- lightning, lightning_lite8, lightning_lite16 \n \
                            \t - thunder, thunder_lite8, thunder_lite16"
            )


def initialize_and_run(infer_model, video_source, host_ip, host_port, debug):
    # Inference model hub
    tensor_hub = retrive_hub(infer_model)
    # VideoPose estimator...
    vp = VideoPose(tensor_hub, source=video_source, debug=debug)
    # UDP Transmitter
    transmitter = QUDPTransmitter(ip=host_ip, port=host_port, debug=debug)

    # #
    # Exit!
    def enter_to_exit():
        print("-" * 100)
        input("Press 'Enter' to exit...\n\n\n")
        vp.stop_processing()
        transmitter.stop_transmission()

    exit_thread = threading.Thread(target=enter_to_exit, daemon=True)
    exit_thread.start()

    transmitter.start_transmission()
    vp.start_processing(callback=transmitter.put_message)


if __name__ == "__main__":
    args = parse_arguments()
    # Calling main function with arguments from command line
    initialize_and_run(args.infer_model, args.source, args.host_ip, args.host_port, args.debug)
