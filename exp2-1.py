# -*-coding: utf-8 -*_
from huaweicloud_sis.client.tts_client import TtsCustomizationClient
from huaweicloud_sis.bean.tts_request import TtsCustomRequest
from huaweicloud_sis.bean.sis_config import SisConfig
from huaweicloud_sis.exception.exceptions import ClientException
from huaweicloud_sis.exception.exceptions import ServerException
import json

ak = "OWZJHZ6MGYURXYVKQOLC"  # 酉配置自己的ak
sk = "m7DqODSvMLGnjIPKqLgb8ugkVBf08hO1SrQps1j4"  # 配置自己的sk
project_id = "52a711036be848569c93f2eb19408eb8"  # 配置自己的project_id
region = "cn-north-4"  # 默认使用北京-4区，对应的区域代码即为cn-north-4

text = 'I am wangjianchi and I come from Jilin University!'  # 待合成文本，不超过500字
path = 'data-2/test.wav'  # 保存路径，可在设置中选择不保存本地

config = SisConfig()
config.set_connect_timeout(5)  # 设置连接超时，单位s
config.set_read_timeout(10)  # 设置读取超时，单位s
ttsc_client = TtsCustomizationClient(ak, sk, region, project_id, sis_config=config)

ttsc_request = TtsCustomRequest(text)
# 设置请求，所有参数均可不设置，使用默认参数
# 设置属性字符串，language_speaker_domain，默认chinese_xiaoyan_common，参考api文档
ttsc_request.set_property('chinese_xiaoyan_common')
# 设置音频格式，默认wav，可选mp3和pcm
ttsc_request.set_audio_format('wav')
# 设置采样率，8000 or 16000，默认8000
ttsc_request.set_sample_rate('8000')
# 设置音量，[0, 100]，默认50
ttsc_request.set_volume(50)
# 设置音高,[-500,500]，默认0
ttsc_request.set_pitch(0)
# 设置音速,[-5O0,500]，默认О
ttsc_request.set_speed(0)
# 设置是否保存，默认False
ttsc_request.set_saved(True)
# 设置保存路径，只有设置保存，此参数才生效
ttsc_request.set_saved_path(path)

# 发送请求，返回结果。如果设置保存，可在指定路径里查看保存的音频。
result = ttsc_client.get_ttsc_response(ttsc_request)
print(json.dumps(result, indent=2, ensure_ascii=False))
