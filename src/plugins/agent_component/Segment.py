import json
import asyncio

from .utils import ComponentABC
from Agently.utils import to_json_desc, find_json

class Segment(ComponentABC):
    def __init__(self, agent: object):
        self.agent = agent
        self.is_enabled = lambda: self.agent.settings.get_trace_back("component_toggles.Segment")
        self.segments = {}
        self.response_segments_cache = {}
        self.parse_stage = 0
        self.current_segment = ""
        self.response_buffer = ""
        self.buffer_next_char = {
            "start": {
                "": "<",
                "<": "!",
                "<!": "%",
                "<!%": "%",
                "<!%%": "=",
            },
            "end": {
                "": "<",
                "<": "/",
                "</": "!",
                "</!": "%",
                "</!%": "%",
                "</!%%": ">",
            },
        }

    def toggle(self, is_enabled: bool):
        self.agent.settings.set("component_toggles.Segment", is_enabled)
        self.agent.refresh_plugins()
        return self.agent   

    def add_segment(self, name: str, output_prompt: any, listener: callable=None, *, is_streaming = False, is_await = False):
        if name not in self.segments:
            self.segments[name] = {
                "prompt": output_prompt,
                "streaming_listeners": [],
                "done_listeners": [],
            }
            if listener != None:
                self.add_segment_listener(name, listener, is_streaming=is_streaming, is_await=is_await)
            return self.agent
        else:
            raise Exception(f"[Agent Component: Segment] segment name '{ name }' has already been used.")

    def add_segment_listener(self, name: str, listener: callable, *, is_streaming = False, is_await = False):
        if name in self.segments:
            if is_streaming:
                self.segments[name]["streaming_listeners"].append((listener, is_await))
            else:
                self.segments[name]["done_listeners"].append((listener, is_await))
        return self.agent

    def on_segment_delta(self, name: str, listener: callable, *, is_await = False):
        return self.add_segment_listener(name, listener, is_streaming = True, is_await= is_await)

    def on_segment_done(self, name: str, listener: callable, *, is_await = False):
        return self.add_segment_listener(name, listener, is_streaming = False, is_await= is_await)

    def _prefix(self):
        if not self.is_enabled() or len(self.segments) == 0:
            return None
        full_output_prompt = "- [OUTPUT RULE]: output content of each segment MUST START WITH TAG:<!%%={{segment_name}}>\n- [REQUIEMENT]:\n"
        for name in self.segments:
            full_output_prompt += f"<!%%={ name }>\n{ to_json_desc(self.segments[name]['prompt']) }\n"
        self.agent.output(full_output_prompt)

    async def _load_json_with_fix(self, origin: str):
        try:
            return json.loads(find_json(origin))
        except json.JSONDecodeError as e:
            try:
                fixed_result = await self.agent.request\
                    .input({
                        "origin JSON String": origin ,
                        "error": e.msg,
                        "position": e.pos,
                    })\
                    .output('Fixed JSON String can be parsed by Python only without explanation and decoration.')\
                    .get_result_async()
                fixed_result = json.loads(find_json(fixed_result))
            except Exception as e:
                raise Exception(f"[Agent Request] Error still occured when try to fix JSON decode error: { str(e) }")
        return fixed_result

    async def _suffix(self, event:str, data: any):
        if not self.is_enabled():
            return None
        if len(self.segments) > 0:
            if event == "response:delta":
                for char in data:
                    '''For Debug
                    print('char', char)
                    print('stage', self.parse_stage)
                    print('buffer', self.response_buffer)
                    print('cache', self.response_segments_cache)
                    print('--------------------')
                    '''
                    #Check Start
                    if self.parse_stage == 0:
                        if self.response_buffer in self.buffer_next_char["start"] and self.buffer_next_char["start"][self.response_buffer] == char:
                            self.response_buffer += char
                        if self.response_buffer == "<!%%=":
                            self.response_buffer = ""
                            self.parse_stage = 1
                            continue
                    #Record Segment Name
                    if self.parse_stage == 1:
                        if char != ">":
                            self.response_buffer += char
                        else:
                            self.current_segment = self.response_buffer
                            self.response_segments_cache[self.current_segment] = ""
                            self.response_buffer = ""
                            self.parse_stage = 2
                            continue
                    #Record Segment Content
                    if self.parse_stage == 2:
                        if self.response_buffer in self.buffer_next_char["start"] and self.buffer_next_char["start"][self.response_buffer] == char:
                            self.response_buffer += char
                        else:
                            self.response_segments_cache[self.current_segment] += self.response_buffer + char
                            self.response_buffer = ""
                            for (streaming_listener, is_await) in self.segments[self.current_segment]["streaming_listeners"]:
                                if asyncio.iscoroutinefunction(streaming_listener):
                                    if is_await:
                                        await streaming_listener(char)
                                    else:
                                        asyncio.create_task(streaming_listener(char))
                                else:
                                    streaming_listener(char)
                        if self.response_buffer == "<!%%=":
                            segment_data = self.response_segments_cache[self.current_segment]
                            if isinstance(self.segments[self.current_segment]["prompt"], (dict, list, set)):
                                segment_data = await self._load_json_with_fix(segment_data)
                            for (done_listener, is_await) in self.segments[self.current_segment]["done_listeners"]:
                                if asyncio.iscoroutinefunction(done_listener):
                                    if is_await:
                                        await done_listener(done_segment_data)
                                    else:
                                        asyncio.create_task(done_listener(segment_data))
                                else:
                                    done_listener(segment_data)
                            self.response_buffer = ""
                            self.current_segment = ""
                            self.parse_stage = 1
            if event == "response:done":
                segment_data = self.response_segments_cache[self.current_segment]
                if isinstance(self.segments[self.current_segment]["prompt"], (dict, list, set)):
                    segment_data = await self._load_json_with_fix(segment_data)
                for (done_listener, is_await) in self.segments[self.current_segment]["done_listeners"]:
                    if asyncio.iscoroutinefunction(done_listener):
                        if is_await:
                            await done_listener(done_segment_data)
                        else:
                            asyncio.create_task(done_listener(segment_data))
                    else:
                        done_listener(segment_data)
                self.agent.request.response_cache["reply"] = self.response_segments_cache
            if event == "response:finally":
                # clean request runtime
                self.segments = {}
                self.response_segments_cache = {}
                self.parse_stage = 0
                self.current_segment = ""
                self.response_buffer = ""

    def export(self):
        return {
            "prefix": self._prefix,
            "suffix": self._suffix,
            "alias": {
                "toggle_segment": { "func": self.toggle },
                "segment": { "func": self.add_segment },
                "add_segment_listener": { "func": self.add_segment_listener },
                "on_segment_delta": { "func": self.on_segment_delta },
                "on_segment_done": { "func": self.on_segment_done },
            },
        }

def export():
    return ("Segment", Segment)