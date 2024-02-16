import threading

import tf_hub_models as th_models
from qudp_transmitter import QUDPTransmitter
from video_pose import VideoPose

MODELS_Movenet_Light = "movenet_lightning"
MODELS_Movenet_Light_Litle_8 = "movenet_lightning_int8.tflite"
MODELS_Movenet_Light_Litle_16 = "movenet_lightning_f16.tflite"
MODELS_Movenet_Thunder = "movenet_thunder"
MODELS_Movenet_Thunder_Litle_8 = "movenet_thunder_int8.tflite"
MODELS_Movenet_Thunder_Litle_16 = "movenet_thunder_f16.tflite"

infer_method, model_input_size = th_models.get_infer_and_input_size(MODELS_Movenet_Thunder_Litle_16)

vp = VideoPose(infer_method, model_input_size)
transmitter = QUDPTransmitter(ip="127.0.0.1", port=4900)

udp_thread = threading.Thread(target=transmitter.activate_transmission)
udp_thread.start()


def enter_to_exit():
    print("-" * 100)
    input("Press 'Enter' to exit...\n\n\n")
    vp.stop_processing()
    transmitter.stop_transmission()


exit_thread = threading.Thread(target=enter_to_exit)
exit_thread.start()

debug = True
if debug:
    vp.start_processing(0, process_pipleline=transmitter.put_message, display_results=True)
else:
    video_thread = threading.Thread(target=vp.start_processing, args=(0, transmitter.put_message, False))
    video_thread.start()
    #    # Esperar a que los threads terminen
    video_thread.join()


udp_thread.join()
exit_thread.join()
