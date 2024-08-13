# consumers.py in your_app
import json
from channels.generic.websocket import AsyncWebsocketConsumer
import asyncio
from dashboard.utilities_helpers.utilities import get_task_position    # import the helper function

class QueuePositionConsumer(AsyncWebsocketConsumer):
    async def connect(self):
        self.task_id = self.scope['url_route']['kwargs']['task_id']
        await self.accept()
        # Start sending position updates periodically
        self.update_task_position_loop = asyncio.ensure_future(self.send_position_updates())

    async def disconnect(self, close_code):
        # Cancel the update loop when the WebSocket connection closes
        if hasattr(self, 'update_task_position_loop'):
            self.update_task_position_loop.cancel()

    async def send_position_updates(self):
        while True:
            position = get_task_position(self.task_id)
            await self.send(text_data=json.dumps({
                'task_id': self.task_id,
                'position': position,
            }))
            await asyncio.sleep(2)  # Adjust the interval as needed

    # Optionally, handle messages from the client if needed
    async def receive(self, text_data):
        pass