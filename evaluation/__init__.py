import logging
import sys

logging.basicConfig(
    format='[LOG]: %(asctime)s,%(msecs)d %(levelname)-8s [%(pathname)s:%(lineno)d in \
    function %(funcName)s] %(message)s',
    datefmt='%Y-%m-%d:%H:%M:%S',
    level=logging.WARN,
    stream=sys.stdout
)