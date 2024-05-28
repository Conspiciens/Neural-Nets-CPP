import socket
import struct
import cv2
import pickle


def client_connect(): 
    client_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    host_ip = '192.168.1.15'
    port = 10050

    # 192.168.1.36

    client_socket.connect((host_ip, port))
    data = b''
    payload_size = struct.calcsize("Q")

    while True:
        while len(data) < payload_size:
            packet = client_socket.recv(4 * 1024)
            if not packet: break
            data += packet

        packed_msg_size = data[:payload_size]
        data = data[payload_size:]
        msg_size = struct.unpack("Q", packed_msg_size)[0]

        while len(data) < msg_size:
            data += client_socket.recv(4 * 1024)

        frame_data = data[:msg_size]
        data = data[msg_size:]
        frame = pickle.loads(frame_data)

        cv2.imshow("Reciving...", frame)
        key = cv2.waitKey(10)
        if key == 13:
            break
    client_socket.close()


def main(): 
    client_connect() 

if __name__ == '__main__': 
    main()
