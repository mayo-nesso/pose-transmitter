import queue
import socket

import msgpack


class QUDPTransmitter:
    def __init__(self, ip, port):
        self.ip = ip
        self.port = port
        self.mqueue = queue.Queue()
        self.sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)

    def _send(self, msge):
        self.sock.sendto(msge, (self.ip, self.port))

    def put_message(self, keypoint_locs, keypoint_edges, edge_colors):
        msge_as_bytes = msgpack.packb({"locs": keypoint_locs.tolist(), "edges": keypoint_edges.tolist()})
        self.mqueue.put(msge_as_bytes)

    def activate_transmission(self):
        while True:
            data = self.mqueue.get()

            if data is None:
                print("EOD: No more data!")
                break
            self._send(data)
