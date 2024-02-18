import threading

from inference_hub import hub
from qudp_transmitter import QUDPTransmitter
from video_pose import VideoPose

debug = True

# Inference model hub
hub = hub.init_lightning_lite8()
# VideoPose estiamtor...
vp = VideoPose(hub, source=0, debug=debug)
# UDP Transmitter
transmitter = QUDPTransmitter(ip="127.0.0.1", port=4900, debug=debug)

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
