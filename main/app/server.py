import socket
import threading

class Socket_server():
    flag = False
    flag_active = False

    def __init__(self):
        t = threading.Thread(target=self.server_pro)
        t.start()

    #porneste serverul
    def server_pro(self):
        """
            Se initializeaza serverul si se asculta dupa o conexiune.
        """
        #obtain host address
        host =socket.gethostname()
        print(host)

        #get instance
        server_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        server_socket.bind(("0.0.0.0", 5001))

        server_socket.listen(1)
        #accept new connection
        conn, address = server_socket.accept()
        #daca se primeste conexiune se va intra in bucla
        print("Connection from: " + str(address))

        #we are using while to run the server and to wait for client request/messages
        while True:
            # daca un flag care e by default setat false -> devine true, se trimite spre client mesaj
            #receive data stream < 1024 bytes
            data = "stop!"
            if self.flag:
                conn.send(data.encode())
                print("Stop")
                self.flag = False

            data1 = "start"
            if self.flag_active:
                conn.send(data1.encode())
                print("Start")
                self.flag_active = False

        #closing the connection
        conn.close()

    def transmit_alert(self, active):
        if active:
            self.flag = True
        else:
            self.flag_active = True

if __name__ == '__main__':
    Socket_server()