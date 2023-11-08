import asyncio
import threading
import json
import websockets
import tornado.ioloop
import tornado.web
import tornado.websocket
import aiohttp
import time
import queue
import uuid
import copy

class WebSocketKeepAlive(object):
    def __init__(self, uri: str, client_id: str):
        self.uri = uri
        self.client_id = client_id
        self.stop_signal = threading.Event()
        self.send_signal = threading.Event()
        self.send_queue = queue.Queue()
        self.event_handlers = {}
        self.thread = None
        self.client = None

        async def send_message(websocket):
            while True:
                while not self.send_signal.is_set():
                    await asyncio.sleep(0)
                try:
                    message = self.send_queue.get_nowait()
                    await websocket.send_str(json.dumps(message))
                except queue.Empty:
                    self.send_signal.clear()
                
        async def receive_message(websocket):
            while True:
                await asyncio.sleep(0)
                if len(self.event_handlers.keys()) > 0:
                    async for message in websocket:
                        message = json.loads(message.data)
                        if message["client_id"] == self.client_id\
                        and message["client_id"] in self.event_handlers\
                        and message["event"] in self.event_handlers[message["client_id"]]:
                            for handler in self.event_handlers[message["client_id"]][message["event"]]:
                                if asyncio.iscoroutinefunction(handler):
                                    await handler(message["data"])
                                else:
                                    handler(message["data"])

        async def stop_keep_alive():
            while not self.stop_signal.is_set():
                await asyncio.sleep(0)
            asyncio.get_event_loop().stop()
            self.client = None

        async def connect_keep_alive():
            async with aiohttp.ClientSession() as session:
                async with session.ws_connect(self.uri) as websocket:
                    self.client = websocket
                    await asyncio.gather(send_message(websocket), receive_message(websocket))

        def start_event_loop():
            asyncio.set_event_loop(asyncio.new_event_loop())
            self.queue = asyncio.Queue()
            asyncio.get_event_loop().run_until_complete(connect_keep_alive())
            asyncio.get_event_loop().run_forever()       

        self.thread = threading.Thread(target=start_event_loop)
        self.thread.start()

        while not self.client:
            time.sleep(0.1)

    def send(self, event: str, data: any=None,):
        request_id = str(uuid.uuid4())
        self.send_queue.put_nowait({ "request_id": request_id, "event": event, "data": data })
        self.send_signal.set()
        return self

    def on(self, event: str, handler: callable):
        if self.client_id not in self.event_handlers:
            self.event_handlers.update({ self.client_id: {} })
        if event not in self.event_handlers[self.client_id]:
            self.event_handlers[self.client_id].update({ event: [handler] })
        else:
            self.event_handlers[self.client_id][event].append(handler)
        return self

    def stop(self):
        self.stop_signal.set()
        self.thread.join()
        self.client = None
        self.queue = None
        self.thread = None
        return self

class WebSocketClient(object):
    def __init__(self, *, host: str="0.0.0.0", port: int=15365, path: str=""):
        self.host = host
        self.port = port
        self.path = path
        self.client_id = str(uuid.uuid4())

    def create_keep_alive(self, path: str=None):
        path = path if path else self.path
        uri = f"ws://{ self.host }:{ self.port }{ path if path[0] == '/' else '/' + path }?client_id={ self.client_id }"
        return WebSocketKeepAlive(uri, self.client_id)

    async def send_async(self, event: str, data: any=None, handler: callable=None, *, path: str=None):
        path = path if path else self.path
        request_id = str(uuid.uuid4())
        uri = f"ws://{ self.host }:{ self.port }{ path if path[0] == '/' else '/' + path }?client_id={ self.client_id }"
        async with aiohttp.ClientSession() as session:
            async with session.ws_connect(uri) as websocket:
                send_message = json.dumps({ "request_id": request_id, "event": event, "data": data })
                await websocket.send_str(send_message)
                if handler != None:
                    try:
                        async for response_message in websocket:
                            response_message = json.loads(response_message.data)
                            if response_message["client_id"] == self.client_id and response_message["request_id"] == request_id:
                                if asyncio.iscoroutinefunction(handler):
                                    await handler(response_message)
                                else:
                                    handler(response_message)
                    except:
                        pass

    def send(self, event: str, data: any=None, handler: callable=None):
        asyncio.get_event_loop().run_until_complete(self.send_async(event, data, handler))
        return self

class WebSocketServer(object):
    def __init__(self, *, host: str="0.0.0.0", port: int=15365):
        self.host = host
        self.port = port
        self.stop_signal = threading.Event()
        self.is_handling = threading.Event()
        self.event_handlers = {}
        self.thread = None
        self.server = None
        self.loop = None
        self.add_handler_queue = queue.Queue()
        self.status = 0

    def set_port(self, port: int):
        self.port = port
        return self

    def start(self, port: int=None):
        def start_loop():
            self.port = port if port else self.port
            self.server = tornado.web.Application([])
            self.server.listen(port)
            if not self.add_handler_queue.empty():
                while True:
                    try:
                        self.add_handler_queue.get_nowait()()
                    except queue.Empty:
                        break
            self.loop = tornado.ioloop.IOLoop.current()
            self.status = 1
            self.loop.start()

        self.thread = threading.Thread(target=start_loop)
        self.thread.start()

        while self.status != 1:
            time.sleep(0.1)

    def stop(self):
        self.loop.stop()
        self.thread.join()
        self.status = 0
        return self

    class WebSocketRequestHandler(tornado.websocket.WebSocketHandler):
        def initialize(self, event_handlers):
            self.event_handlers = event_handlers

        def _response_message(self, event: str, data: any=None, *, close_after_response: bool=False):
            self.write_message(json.dumps({
                "client_id": self.client_id,
                "request_id": self.request_id,
                "event": event,
                "data": data,
            }))
            if close_after_response:
                self.close()
            return self.close

        async def on_message(self, message: str):
            message = json.loads(message)
            self.client_id = self.get_argument("client_id")
            self.request_id = message["request_id"]
            if message["event"] in self.event_handlers:
                for handler in self.event_handlers[message["event"]]:
                    if asyncio.iscoroutinefunction(handler):
                        await handler(message["data"], self._response_message)
                    else:
                        handler(message["data"], self._response_message)

    def add_event_handler(self, path: str, event: str, handler: callable):
        if path == None or path == "" or path == "/":
            raise Exception("[WebSocket Server] Event handler require param 'path' that is not '','/' or None.")
        if path[0] != "/":
            path = "/" + path
        if path in self.event_handlers:
            if event in self.event_handlers[path]:
                self.event_handlers[path][event].append(handler)
            else:
                self.event_handlers[path].update({ event: [handler] })
        else:
            self.event_handlers.update({ path: { event: [handler] } })
            if self.server:
                while True:
                    try:
                        self.add_handler_queue.get_nowait()()
                    except queue.Empty:
                        break
                self.server.add_handlers(".*$", [(path, self.WebSocketRequestHandler, { "event_handlers": self.event_handlers[path] })])
            else:
                self.add_handler_queue.put_nowait(lambda: self.server.add_handlers(".*$", [(path, self.WebSocketRequestHandler, { "event_handlers": self.event_handlers[path] })]))
        return self

    def remove_event_handler(self, path:str, event: str=None):
        if event == None:
            if path in self.event_handlers:
                del self.event_handlers[path]
        else:
            if path in self.event_handlers and event in self.event_handlers[path]:
                del self.event_handlers[path][event]
        self.server.remove_handlers(path)

    def __exit__(self):
        self.stop()