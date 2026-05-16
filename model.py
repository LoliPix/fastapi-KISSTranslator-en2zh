from pydantic import BaseModel, ConfigDict, Field


# 只支持聚合翻译
class TranslationRequest(BaseModel):
    texts: list[str]
    from_lang: str = Field(alias='from',description='源语言只支持英文')
    to_lang: str = Field(alias='to',description='目标语言只支持简体中文')
    model_config = ConfigDict(populate_by_name=True)    # 允许设置别名

class TranslationItem(BaseModel):
    text: str = Field(description='译文只支持简体中文')
    src: str = Field(default='en',description='源语言只支持英文')

class TranslationResponse(BaseModel):
    translations: list[TranslationItem]
