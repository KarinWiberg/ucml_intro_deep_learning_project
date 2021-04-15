{
 "cells": [
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "outputs": [],
   "source": [
    "import signal\n",
    "\n",
    "from contextlib import contextmanager\n",
    "\n",
    "import requests\n",
    "\n",
    "\n",
    "DELAY = INTERVAL = 4 * 60  # interval time in seconds\n",
    "MIN_DELAY = MIN_INTERVAL = 2 * 60\n",
    "KEEPALIVE_URL = \"https://nebula.udacity.com/api/v1/remote/keep-alive\"\n",
    "TOKEN_URL = \"http://metadata.google.internal/computeMetadata/v1/instance/attributes/keep_alive_token\"\n",
    "TOKEN_HEADERS = {\"Metadata-Flavor\":\"Google\"}\n",
    "\n",
    "\n",
    "def _request_handler(headers):\n",
    "    def _handler(signum, frame):\n",
    "        requests.request(\"POST\", KEEPALIVE_URL, headers=headers)\n",
    "    return _handler\n",
    "\n",
    "\n",
    "@contextmanager\n",
    "def active_session(delay=DELAY, interval=INTERVAL):\n",
    "    \"\"\"\n",
    "    Example:\n",
    "\n",
    "    from workspace_utils import active session\n",
    "\n",
    "    with active_session():\n",
    "        # do long-running work here\n",
    "    \"\"\"\n",
    "    token = requests.request(\"GET\", TOKEN_URL, headers=TOKEN_HEADERS).text\n",
    "    headers = {'Authorization': \"STAR \" + token}\n",
    "    delay = max(delay, MIN_DELAY)\n",
    "    interval = max(interval, MIN_INTERVAL)\n",
    "    original_handler = signal.getsignal(signal.SIGALRM)\n",
    "    try:\n",
    "        signal.signal(signal.SIGALRM, _request_handler(headers))\n",
    "        signal.setitimer(signal.ITIMER_REAL, delay, interval)\n",
    "        yield\n",
    "    finally:\n",
    "        signal.signal(signal.SIGALRM, original_handler)\n",
    "        signal.setitimer(signal.ITIMER_REAL, 0)\n",
    "\n",
    "\n",
    "def keep_awake(iterable, delay=DELAY, interval=INTERVAL):\n",
    "    \"\"\"\n",
    "    Example:\n",
    "\n",
    "    from workspace_utils import keep_awake\n",
    "\n",
    "    for i in keep_awake(range(5)):\n",
    "        # do iteration with lots of work here\n",
    "    \"\"\"\n",
    "    with active_session(delay, interval): yield from iterable\n",
    "\n"
   ]
  }
 ],
 "metadata": {
  "kernelspec": {
   "display_name": "Python 3",
   "language": "python",
   "name": "python3"
  },
  "language_info": {
   "codemirror_mode": {
    "name": "ipython",
    "version": 3
   },
   "file_extension": ".py",
   "mimetype": "text/x-python",
   "name": "python",
   "nbconvert_exporter": "python",
   "pygments_lexer": "ipython3",
   "version": "3.8.5"
  }
 },
 "nbformat": 4,
 "nbformat_minor": 4
}
