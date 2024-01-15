import socket
import struct 
import time 
import contextlib 
import threading 
import functools 
import queue 

import numpy as np 

    
class LEDClient:

    def __init__(
        self,
        host, 
        port=80,
        nled=100,
        byte_order='<',  #Little endian (<), see: https://docs.python.org/3/library/struct.html
        sleep_ms=10,
    ):
        self.host = host 
        self.port = port 
        self.nled = nled 
        self.byte_order = byte_order 
        self.sleep_ms = sleep_ms

        self._sock = None 
        self._data = np.zeros((nled, 3), dtype=np.uint8) 
        self._buff = queue.Queue(20)
        self._listener = None 

    @property 
    def connected(self):
        return self._sock is not None 

    def connect(self):
        assert not self.connected 
        self._sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self._sock.connect((self.host, self.port))
        self._listener = threading.Thread(target=self.listen)
        self._listener.start()
        
    def disconnect(self):
        assert self.connected 
        while not self._buff.empty():
            try:
                self._buff.get_nowait()
            except: pass
        self._buff.put(None)
        self._listener.join()
        self._sock.close()
        self._sock = None 
        print("Waiting for listener...")
        
    def __enter__(self):
        assert not self.connected 
        self.connect()
        return self 
    
    def __exit__(self, type, value, traceback):
        if self.connected:
            self.disconnect()

    def __len__(self):
        return len(self._data)
    
    @contextlib.contextmanager
    def maybe_connect(self):
        if self.connected:
            yield
        else:
            with self:
                yield 


    @contextlib.contextmanager 
    def update(self):
        leds = self._data.copy()
        yield leds 
        self._data[:] = leds 
        self.show()

    def show(self):
        with self.maybe_connect():
            self.send(
                1,
                self.dense_header(0, len(self), 1),
                np.ascontiguousarray(self._data, f'{self.byte_order}B').tobytes(),
            )

    def send(self, mode, head, body):
        assert self.connected
        # if self._buff.full():
        #     print("WARNING: QUEUE FULL!")
        self._buff.put(time.perf_counter())
        self._sock.sendall(bytes([99, 99, 99, mode])) # Preamble
        self._sock.sendall(head)
        self._sock.sendall(body)
        if self.sleep_ms:
            time.sleep(max(0, self.sleep_ms/1000))

    def dense_header(self, offset, length, stride):
        return self.pack('HHH', offset, length, stride)
    
    def pack(self, s, *vars):
        return struct.pack(self.byte_order + s, *vars)

    def listen(self):
        assert self.connected 
        while True:
            item = self._buff.get()
            if item is None:
                print("Stopping listener")
                return
            while True:
                resp = self._sock.recv(1)
                if len(resp):
                    break
            try:
                # print('Response!', int(resp))
                pass
            except Exception as e:
                print('Error', e, 'when reading resp', str(resp))
                raise e


def up_down_sequence(lo, hi, phase=0):
    assert lo < hi 
    period = hi - lo 
    vel = +1 if (phase / period) % 2 == 0 else -1
    i = phase % period 
    while True:
        yield lo + i 
        i = i + vel 
        if i == period:
            i = period - 1 
            vel = -1 
        if i < 0:
            i = 0
            vel = +1 

def main1():

    with LEDClient('192.168.4.1', nled=400) as client:

        print('Connected')
        pos = 0
        vel = 1
        while True:
            for i in range(800):
                t0 = time.perf_counter()
                with client.update() as leds:
                    v = min(399, 400 - int(abs(i - 400)))
                    # print(v)
                    leds[:, 0] = (100 - v/4) % 200
                    leds[:, 1] = (v/4) % 200
                    leds[:, 2] = 0
                    leds[v] = (0, 0, 255)
                # print(i)
                # time.sleep(1/60)
            print('POP')

def main2():
    delta = 10
    with LEDClient('192.168.4.1', nled=400) as client:
        with client.update() as leds:
            # leds[:] = (100, 0, 255)
            # leds[:] = (255, 100, 0)
            # leds[:] = (0, 200, 0)
            leds[:] = (255, 255, 255)
        # time.sleep(3)
        # while True:
        #     with client.update() as leds:
        #         data = leds.astype(int)
        #         data += np.random.randint(-delta, delta+1, size=data.shape)
        #         leds[:] = data.clip(0, 255)

def main3():
    with LEDClient('192.168.4.1', nled=400) as client:
        while True:
            for i in range(3):
                with client.update() as leds:
                    leds[:] = 0
                    leds[:, i] = 255



def zigzagseq(lo, hi, phase=0):
    assert lo < hi 
    period = hi - lo 
    vel = +1 if (phase / period) % 2 == 0 else -1
    i = phase % period 
    while True:
        yield lo + i 
        i = i + vel 
        if i == period:
            i = period - 1 
            vel = -1 
        if i < 0:
            i = 0
            vel = +1 

def main4():
    w = 5
    nled = 400 
    seqr = zigzagseq(100, 200)
    seqg = zigzagseq(0, 100, 100)
    seqb = zigzagseq(-w//2, nled + w//2, 0)
    with LEDClient('192.168.4.1', nled=nled) as client:
        while True:
            for i in range(3):
                with client.update() as leds:
                    leds[:, 0] = next(seqr)
                    leds[:, 1] = next(seqg)
                    i = next(seqb)
                    leds[:, 2] = 0
                    for di in range(-w//2, w//2):
                        j = i + di 
                        if j < 0 or j >= nled:
                            continue 
                        v = int(255 * np.exp(-di ** 2 / (0.5 * w)))
                        v = 255
                        assert 0 <= v <= 255, v
                        leds[j, 2] = v 


def main5():
    client = LEDClient('192.168.4.1', nled=400)
    print('Before')
    with client.update() as leds:
        leds[:] = 0 
        leds[:, 1] = 200
    print('After')

def main6():
    with LEDClient('192.168.4.1', nled=400) as client:
        vals = np.full((client.nled, 3), 150, dtype=float)
        while True:
            with client.update() as leds:
                vals = np.clip(
                    vals + np.random.normal(0, 10, size=vals.shape), 
                    (100, 10, 0), 
                    (220, 50, 50)
                )
                leds[:] = vals.astype(int)
                # print('OK')
                # leds[:, 1] = 0
                # leds[:, 2] = 0


    

if __name__ == '__main__':
    t_start = time.time()
    try:
        # main1()
        # main2()
        # main3()
        # main4()#
        # main5()
        main6()
    finally:
        print(f'Finished in {time.time() - t_start :.1f} seconds.')
