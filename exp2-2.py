# -*-coding: utf-8 -*-
from huaweicloud_sis.client.asr_client import AsrCustomizationClient
from huaweicloud_sis.bean.asr_request import AsrCustomShortRequest
from huaweicloud_sis.bean.asr_request import AsrCustomLongRequest
from huaweicloud_sis.exception.exceptions import ClientException
from huaweicloud_sis.exception.exceptions import ServerException
from huaweicloud_sis.utils import io_utils
from huaweicloud_sis.bean.sis_config import SisConfig
import json

ak = "OWZJHZ6MGYURXYVKQOLC"  # 酉配置自己的ak
sk = "m7DqODSvMLGnjIPKqLgb8ugkVBf08hO1SrQps1j4"  # 配置自己的sk
project_id = "52a711036be848569c93f2eb19408eb8"  # 配置自己的project_id
region = "cn-north-4"  # 默认使用北京-4区，对应的区域代码即为cn-north-4

# 一句话识别参数，我们使用语音合成的语音数据，1min 以内的音频
path = 'data-2/test.wav'
path_audio_format = 'wav'  # 音频格式，详见 api文档
path_property = 'chinese_8k_common'  # language_sampleRate_domain，如chinese_8k_common，详见api文档

config = SisConfig()
config.set_connect_timeout(5)  # 设置连接超时
config.set_read_timeout(10)  # 设置读取超时
asr_client = AsrCustomizationClient(ak, sk, region, project_id, sis_config=config)  # 初始化客户端

data = io_utils.encode_file(path)
asr_request = AsrCustomShortRequest(path_audio_format, path_property, data)
# 所有参数均可不设置，使用默认值
# 设置是否添加标点，yes or no，默认no
asr_request.set_add_punc('yes')

# 发送请求，返回结果,返回结果为json格式
result = asr_client.get_short_response(asr_request)
print(json.dumps(result, indent=2, ensure_ascii=False))
