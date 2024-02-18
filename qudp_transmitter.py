import queue
import socket
import threading
from time import time

import msgpack


class QUDPTransmitter:

    def __init__(self, ip, port, debug=False):
        self.ip = ip
        self.port = port
        self.mqueue = queue.Queue()
        self.sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        self.udp_thread = threading.Thread(target=self._activate_transmission, daemon=True)
        self._stop_event = threading.Event()
        self.start_time = time()
        self.DEBUG = debug

    def _send(self, msge):
        self.sock.sendto(msge, (self.ip, self.port))
        if self.DEBUG:
            # FPS
            end_time = time()
            fps = 1 / (end_time - self.start_time)
            print(f"QUDPTransmitter: _send FPS: {fps:.1f}")
        self.start_time = end_time

    def put_message(self, keypoint_locs, keypoint_edges, edge_colors):
        msge = {"locs": keypoint_locs.tolist(), "edges": keypoint_edges.tolist()}
        if self.DEBUG:
            print("New Msge Queued: - - - - -:")
            print(keypoint_locs.tolist())

        msge_as_bytes = msgpack.packb(msge)
        self.mqueue.put(msge_as_bytes)

    def _activate_transmission(self):
        while not self._stop_event.is_set():
            data = self.mqueue.get()
            self._send(data)

    def start_transmission(self):
        self.udp_thread.start()

    def stop_transmission(self):
        self._stop_event.set()
