### setup all path variables and logging
import sys
import os
import logging

logging.basicConfig(level=logging.INFO)

sys.path.append(os.path.join(os.path.abspath('../..'),'information_bottleneck','alternating_information_bottleneck_algorithms'))
sys.path.append(os.path.join(os.path.abspath('../..'),'information_bottleneck','alternating_information_bottleneck_algorithms','generic_classes'))
sys.path.append(os.path.join(os.path.abspath('../..'),'information_bottleneck','alternating_information_bottleneck_algorithms','scalar_IB'))
sys.path.append(os.path.join(os.path.abspath('../..'),'information_bottleneck','alternating_information_bottleneck_algorithms','sIB'))
sys.path.append(os.path.join(os.path.abspath('../..'),'information_bottleneck','ib_base','information_bottleneck_algorithms'))
sys.path.append(os.path.join(os.path.abspath('../..'),'information_bottleneck','ib_base','tools'))


