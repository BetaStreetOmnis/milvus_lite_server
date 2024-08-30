# common_logic.py
from fastapi import HTTPException
from fastapi.encoders import jsonable_encoder
import traceback


"""通过的fastapi请求方式"""
def common_logic(item, common_fun):
    try:
        item = jsonable_encoder(item)
        res = common_fun(item)
        return {"status_code": 200,
                "data": res
                }
    except Exception as e:
        print(e)
        print(traceback.format_exc())
        raise HTTPException(status_code=430, detail=str(traceback.format_exc))
