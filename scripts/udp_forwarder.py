"""
UDP Forwarder — run on the Laptop (Windows).

Quest sends hand-tracking UDP packets to this Laptop via USB Link.
This script receives them and forwards to the PC running human_data_collection.py.

Usage:
    python scripts/udp_forwarder.py --target_ip 192.168.9.155 --port 12346
"""

import socket
import argparse
import sys

def forward(listen_port: int, target_ip: str, target_port: int):
    recv_sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    recv_sock.bind(("", listen_port))
    recv_sock.setblocking(True)

    send_sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)

    print(f"Forwarding UDP :{listen_port} -> {target_ip}:{target_port}")
    print("Press Ctrl+C to stop.\n")

    try:
        while True:
            data, addr = recv_sock.recvfrom(8192)
            send_sock.sendto(data, (target_ip, target_port))
    except KeyboardInterrupt:
        print("Forwarder stopped.")
    finally:
        recv_sock.close()
        send_sock.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--target_ip", type=str, default="192.168.9.155",
                        help="IP of the PC running human_data_collection.py")
    parser.add_argument("--port", type=int, default=12346,
                        help="UDP port (must match POSE_INFO_PORT in ip_config.py)")
    args = parser.parse_args()
    forward(listen_port=args.port, target_ip=args.target_ip, target_port=args.port)
