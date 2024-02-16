import queue
import socket

import msgpack


class QUDPTransmitter:
    def __init__(self, ip, port):
        self.ip = ip
        self.port = port
        self.mqueue = queue.Queue()
        self.sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        self.tranmitting = False

    def _send(self, msge):
        self.sock.sendto(msge, (self.ip, self.port))

    def put_message(self, keypoint_locs, keypoint_edges, edge_colors):
        msge_as_bytes = msgpack.packb({"locs": keypoint_locs.tolist(), "edges": keypoint_edges.tolist()})
        self.mqueue.put(msge_as_bytes)

    def activate_transmission(self):
        self.tranmitting = True
        while self.tranmitting:
            data = self.mqueue.get()
            self._send(data)

    def stop_transmission(self):
        self.tranmitting = False
