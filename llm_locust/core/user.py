"""
User simulator for generating load against LLM endpoints.
"""

import asyncio
import logging
import time
from multiprocessing import Queue
from typing import TYPE_CHECKING

import aiohttp

from llm_locust.core.models import (
    ErrorLog,
    RequestFailureLog,
    RequestSuccessLog,
    get_timestamp_seconds,
)

if TYPE_CHECKING:
    from llm_locust.clients.openai import BaseModelClient

logger = logging.getLogger(__name__)


class User:
    """User requests to a model client and sends request info to metrics_queue"""

    def __init__(
        self,
        model_client: "BaseModelClient",
        metrics_queue: Queue,
        user_id: int = 0,
    ) -> None:
        self.metrics_queue = metrics_queue
        self.model_client = model_client
        self.run = True
        self.user_id = user_id
        self._stop_event = asyncio.Event()
        self.start()

    async def stop(self) -> None:
        """Stops the User from sending further requests"""
        self.run = False
        await self._stop_event.wait()  # Wait until user_loop finishes

    def start(self) -> None:
        """Starts the user loop as an asyncio task"""
        asyncio.create_task(self.user_loop())

    async def user_loop(self) -> None:
        """sends request continuously while sending request info to metrics queue"""
        async with aiohttp.ClientSession() as session:
            while self.run:
                url, headers, data, input_data = self.model_client.get_request_params()
                start_time = time.perf_counter()
                try:
                    async with session.post(
                        url,
                        headers=headers,
                        json=data,
                    ) as response:
                        if response.status != 200:
                            # Request is unsuccessful. collect and continue
                            time_key = get_timestamp_seconds()
                            end_time = time.perf_counter()
                            self.metrics_queue.put(
                                RequestFailureLog(
                                    timestamp=time_key,
                                    start_time=start_time,
                                    end_time=end_time,
                                    status_code=response.status,
                                )
                            )
                            error_text = await response.text()
                            logger.info(
                                "Request failure",
                                extra={
                                    "user_id": self.user_id,
                                    "status": response.status,
                                    "error": error_text[:200],
                                },
                            )
                            continue

                        result_chunks: list[bytes] = []
                        token_times: list[float] = []
                        try:
                            async for data, _ in response.content.iter_chunks():
                                token_times.append(time.perf_counter())
                                result_chunks.append(data)
                        except Exception as e:
                            logger.warning(
                                "Error streaming response",
                                exc_info=e,
                                extra={"user_id": self.user_id},
                            )
                            self.metrics_queue.put(
                                ErrorLog(
                                    error_message=str(e),
                                    error_type=type(e).__name__,
                                    context={"user_id": self.user_id},
                                )
                            )
                            continue

                        # Successfully received response
                        time_key = get_timestamp_seconds()
                        end_time = time.perf_counter()
                        self.metrics_queue.put(
                            RequestSuccessLog(
                                result_chunks=tuple(result_chunks),
                                num_input_tokens=input_data["num_input_tokens"],
                                timestamp=time_key,
                                token_times=tuple(token_times),
                                start_time=start_time,
                                end_time=end_time,
                                status_code=response.status,
                            )
                        )
                except Exception as e:
                    logger.exception(
                        "Unexpected error in user loop",
                        extra={"user_id": self.user_id},
                    )
                    self.metrics_queue.put(
                        ErrorLog(
                            error_message=str(e),
                            error_type=type(e).__name__,
                            context={"user_id": self.user_id},
                        )
                    )
        self._stop_event.set()
