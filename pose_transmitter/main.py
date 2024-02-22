import argparse
import threading
from dataclasses import dataclass

from inference_hub import hub
from qudp_transmitter import QUDPTransmitter
from video_pose_processor import VideoPoseProcessor


@dataclass
class Configuration:
    video_source: str
    host_ip: str
    host_port: int
    infer_model: str
    debug: bool


def parse_arguments():
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(description="Description of your program")
    parser.add_argument("--video_source", default=0, help="Source file for video input, or 0 for webcam")
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
    parser.add_argument("--debug", action="store_true", help="Enable debug mode")
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


def initialize_objects(config: Configuration):
    tensor_hub = retrive_hub(config.infer_model)
    vp = VideoPoseProcessor(tensor_hub, source=config.video_source, debug=config.debug)
    transmitter = QUDPTransmitter(ip=config.host_ip, port=config.host_port, debug=config.debug)
    return vp, transmitter


def run_processing(vpp, transmitter):
    # #
    # Exit!
    def enter_to_exit():
        print("-" * 100)
        input("Press 'Enter' to exit...\n\n\n")
        vpp.stop_processing()
        transmitter.stop_transmission()

    exit_thread = threading.Thread(target=enter_to_exit, daemon=True)
    exit_thread.start()

    # Start transmission and pose inference!
    vpp.start_processing(callback=transmitter.put_message)
    transmitter.start_transmission()


def main():
    args = parse_arguments()
    vp, transmitter = initialize_objects(args)
    run_processing(vp, transmitter)


if __name__ == "__main__":
    main()
