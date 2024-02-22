import queue
import socket
import threading
from time import time

import msgpack


class QUDPTransmitter:
    """
    Class for transmitting messages via UDP using a queue.
    """
    def __init__(self, ip, port, debug=False):
        """
        Initialize the QUDPTransmitter.

        Args:
            ip (str): IP address to send messages to.
            port (int): Port number to send messages to.
            debug (bool, optional): Flag to enable debug mode. Defaults to False.
        """
        self.ip = ip
        self.port = port
        self.mqueue = queue.Queue()
        self.sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        self.udp_thread = threading.Thread(target=self._activate_transmission, daemon=True)
        self._stop_event = threading.Event()
        self.start_time = time()
        self.debug = debug

    def _send(self, msge):
        """
        Send message via UDP.

        Args:
            message (bytes): Message to send.
        """
        self.sock.sendto(msge, (self.ip, self.port))
        if self.debug:
            # Calculate and print FPS
            end_time = time()
            fps = 1 / (end_time - self.start_time)
            print(f"QUDPTransmitter: _send FPS: {fps:.1f}")
            self.start_time = end_time

    def _activate_transmission(self):
        """
        Start transmitting messages from the queue.
        """
        while not self._stop_event.is_set():
            data = self.mqueue.get()
            self._send(data)

    def put_message(self, keypoint_locs, keypoint_edges, edge_colors):
        """
        Add message to the queue.

        Args:
            keypoint_locs (numpy.ndarray): Keypoint locations.
            keypoint_edges (numpy.ndarray): Keypoint edges.
            edge_colors (list): List of edge colors.
        """
        msge = {"locs": keypoint_locs.tolist(), "edges": keypoint_edges.tolist()}
        if self.debug:
            print("New Msge Queued:")
            print(keypoint_locs.tolist())

        msge_as_bytes = msgpack.packb(msge)
        self.mqueue.put(msge_as_bytes)

    def start_transmission(self):
        """Start message transmission."""
        self.udp_thread.start()

    def stop_transmission(self):
        """Stop message transmission."""
        self._stop_event.set()
