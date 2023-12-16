#  vim: set foldmethod=indent foldcolumn=4 :
#!/usr/bin/env python3

import numpy as np
import pandas as pd

from IPython.display import display_html
import re
import inspect
import textwrap


def color_negative_red(val):
    color = "red" if val < 0 else "black"
    return "color: %s" % color


def highlight_maxFcn(s, props="color:black;background-color:gold"):
    return np.where(s == np.nanmax(s.values), props, "")


def is_html(string):
    if not isinstance(string, str):
        return None

    # Regular expression pattern to match HTML tags
    pattern = r"<[^>]+>"

    # Check if the string contains HTML tags
    match = re.search(pattern, string)
    return match is not None


def getNameOfArgs(*_args):
    caller_frame = inspect.stack()[1][0]
    local_vars = caller_frame.f_locals

    arg_names = []
    for arg in _args:
        if isinstance(arg, str):
            arg_names.append(arg)
        else:
            for key, value in local_vars.items():
                if value is arg:
                    arg_names.append(key)
                    break
            else:
                arg_names.append("")
    return arg_names


def display_htmlFcn(
    *_args,
    _localVars: dict = {},  # locals() # if display_htmlFcn imported
    _localVarsKeys: list = [],  # ["df", "df2"] # or whatever # if locals() have heavy arguments
    _space=0,
    _n=99,
    _float_prec=2,
    _formatVal=None,
    _applyOrApplyMap=None,
    _hide_index=False,
    _hide_columns=False,
    _addStr=None,
    _debugBool=False,
    _returnNone=True,
):
    """Display HTML representation of multiple objects"""

    if (_localVars) and (_localVarsKeys):
        _localVars = {key: _localVars[key] for key in _localVarsKeys}

    if _localVars:
        for _key in _localVars:
            try:
                # https://stackoverflow.com/questions/23168282/setting-variables-with-exec-inside-a-function
                _str01 = f"global {_key}; {_key} = _localVars.get('{_key}')"
                if _debugBool:
                    print(_str01)
                exec(_str01)
            except Exception as err:
                if _debugBool:
                    print(f"\tError2: {err}" + "\n" + 80 * "~")

    if not len(_args):
        return None

    # to not need to quote variables, just directly pass
    if False:
        # not working because local variables of current function are not same scope in getNameOfArgs()
        _arg_names = getNameOfArgs(*_args)
    else:
        # just show name of variables in caption, rest "". Improve this is megahard for the AI, I think is imposible
        # df = pd.DataFrame({'A': rng.rand(5),'B': rng.rand(5)});
        # display_htmlFcn(df.mean(axis=0),df,debugBool=1) # captions: "" and "df"
        # display_htmlFcn("df.mean(axis=0)",df,debugBool=1) # captions: "df.mean(axis=0)" and "df"
        _caller_frame = inspect.stack()[1][0]
        _local_vars = _caller_frame.f_locals

        _arg_names = []
        for _arg in _args:
            if isinstance(_arg, str):
                _arg_names.append(_arg)
            else:
                for _key, _value in _local_vars.items():
                    if _value is _arg:
                        _arg_names.append(_key)
                        break
                else:
                    _arg_names.append("")
    if _debugBool:
        print(f"\t{'arg_names':20s}: {_arg_names}")

    if not isinstance(_hide_index, list):
        _hide_index = np.full((len(_args),), _hide_index)
    if _debugBool:
        print(f"{'_hide_index':20s}: {_hide_index}")

    if not isinstance(_hide_columns, list):
        _hide_columns = np.full((len(_args),), _hide_columns)
    if _debugBool:
        print(f"{'_hide_columns':20s}: {_hide_columns}")

    if not isinstance(_addStr, list):
        _addStr = np.full((len(_args),), _addStr)
    if _debugBool:
        print(f"{'_addStr':20s}: {_addStr}")

    _html_str = ""
    for _index, _k in enumerate(_args):
        if _debugBool:
            print("\n" + 40 * "- " + "\nk:", _k)
            print(f"\t{'type(k)':20s}: {type(_k)}")

        if is_html(_k):  # already rendered pd.DataFrame
            _html_str += _k
            _html_str += int(_space) * "&nbsp"
            continue  # jump to next loop iter

        if isinstance(_k, str):
            __df = eval(
                _k
            )  # left of "= eval(k)" should be something weird, otherwise if it were "df" (df = eval(k))
            # ...then code like next would corrupt in the eval:
            # df = pd.DataFrame({'A': rng.rand(5),'B': rng.rand(5)}); dfMeanAxis0=df.mean(axis=0);
            # display_htmlFcn('df', 'df.mean()',"df.mean(axis=0)","pd.DataFrame(df.mean(axis=0))","dfMeanAxis0",debugBool=1)
        else:
            # to pass dataframes, like: df=make_df('AB',[1,2]);dfArr=[df,df];display_htmlFcn(*dfArr)
            __df = _k

        if _debugBool:
            print(f"\t{'k':20s}: {_k}")
            print(f"\t{'type(__df)':20s}: {type(__df)}")
            print(
                f"\t{'isinstance(__df, pd.DataFrame)':20s}: {isinstance(__df, pd.DataFrame)}"
            )
        if not isinstance(__df, pd.DataFrame):
            __df = pd.DataFrame(__df)
        if _debugBool:
            print(f"\t{'__df.values':20s}: {__df.values}")
            print(f"\t{'__df.columns':20s}: {__df.columns}")

        if isinstance(_k, str):
            if False:
                _caption = [_k[_i : _i + _n] for _i in range(0, len(_k), _n)]
                for i in range(len(_caption)):
                    _caption[i] += "<br>"  # "\n"
            else:
                _caption = textwrap.wrap(_k, width=_n)
                _caption = "<br>".join(_caption)
        else:
            _caption = _arg_names[_index]
        if _debugBool:
            print(f"\t{'_caption':20s}: {_caption}")

        if isinstance(_float_prec, list):
            try:
                _float_precCurrent = _float_prec[_index]
            except:
                _float_precCurrent = _float_prec[-1]  # if few, then get last given
        else:
            _float_precCurrent = _float_prec
        if _debugBool:
            print(f"{'_float_precCurrent':20s}: {_float_precCurrent}")

        def _defaultFormatVal(_float_precCurrent=2):
            if _float_precCurrent is None:
                _float_precCurrent = 2 # fixed
            return "{:." + str(_float_precCurrent) + "f}"  # example formatVal="{:.2f}"
        if isinstance(_formatVal, list):
            try:
                _formatValCurrent = _formatVal[_index]
            except:
                _formatValCurrent = _formatVal[-1]  # if few, then get last given
        else:
            _formatValCurrent = _formatVal
        if _formatValCurrent is None:
            _formatValCurrent = _defaultFormatVal(_float_precCurrent)
        if _debugBool:
            print(f"{'_formatValCurrent':20s}: {_formatValCurrent}")

        if isinstance(_applyOrApplyMap, list):
            try:
                _applyOrApplyMapCurrent = _applyOrApplyMap[_index]
            except:
                _applyOrApplyMapCurrent = _applyOrApplyMap[
                    -1
                ]  # if few, then get last given
        else:
            _applyOrApplyMapCurrent = _applyOrApplyMap

        _applyBool = False
        _applymapBool = False  # default one
        _applyVal = None
        _applymapVal = None
        if isinstance(_applyOrApplyMapCurrent, dict):
            key = "apply"
            if key in _applyOrApplyMapCurrent.keys():
                _applyBool = True
                _applyVal = _applyOrApplyMapCurrent[key]
            key = "applymap"
            if key in _applyOrApplyMapCurrent.keys():
                _applymapBool = True
                _applymapVal = _applyOrApplyMapCurrent[key]
        else:
            _applymapBool = True  # default one
            _applymapVal = _applyOrApplyMapCurrent

        if not isinstance(_applyVal, dict):
            _applyVal = dict(func=_applyVal)
        if not isinstance(_applymapVal, dict):
            _applymapVal = dict(func=_applymapVal)

        if _debugBool:
            print(f"{'_applyOrApplyMapCurrent':23s}: ", _applyOrApplyMapCurrent)
            print(f"{'_applyBool':20s}: {_applyBool}")
            print(f"{'_applymapBool':20s}: {_applymapBool}")
            print(f"{'_applyVal':20s}: {_applyVal}")
            print(f"{'_applymapVal':20s}: {_applymapVal}")

        try:
            _result = (
                __df.style.set_table_attributes("style='display:inline'")
                .set_caption("".join(_caption))
                .format(_formatValCurrent)
            )
            if _applyBool and _applyVal["func"] is not None:
                _result = _result.apply(**_applyVal)
            if _applymapBool and _applymapVal["func"] is not None:
                _result = _result.applymap(**_applymapVal)
            if _hide_index[_index]:
                _result = _result.hide(axis="index")
                #  _result = _result.hide_index() # depreciated
            if _hide_columns[_index]:
                _result = _result.hide(axis="columns")
            if isinstance(_addStr[_index], str):
                _result = eval("_result" + _addStr[_index])
            _html_str += _result._repr_html_()
        except Exception as err:
            if _debugBool:
                print(f"\tError2: {err}" + "\n" + 80 * "~")
            _html_str += (
                __df.style.set_table_attributes("style='display:inline'")
                .set_caption("".join(_caption))
                ._repr_html_()
            )
        _html_str += int(_space) * "&nbsp"

    display_html(_html_str, raw=True)

    if not _returnNone:
        return _html_str
    return None


def test01():
    df = pd.DataFrame(
        np.array([[1, -5, 22, -2, 9], list("ABCBA")]).T, columns=["int1", "str1"]
    )

    dummyCol = "str1"
    # df.drop(columns=[dummyCol], inplace=True)

    html_content = display_htmlFcn(
        "df",
        "df.join(pd.get_dummies(df[dummyCol])).drop(dummyCol,axis=1)",
        _localVars=locals(),
        _localVarsKeys=["df", "dummyCol"],  # or just ommit
        _space=5,
        _n=9999,
        _returnNone=False,
        #  _debugBool=1
    )
    return html_content


def test02():
    df = pd.DataFrame(np.random.randint(0, 100, size=(5, 4)) - 50, columns=list("ABCD"))

    html_content = display_htmlFcn(
        "df",
        "df",
        "df",
        _localVars=locals(),
        _applyOrApplyMap=[
            {"applymap": color_negative_red},
            {"apply": highlight_maxFcn},
            {"applymap": color_negative_red, "apply": highlight_maxFcn},
        ],
        _returnNone=False,
        _debugBool=1,
    )
    return html_content


def test03(returnNone=False):
    c = color_negative_red
    applyArgs = dict(
        func=highlight_maxFcn, props="color:white;background-color:darkblue", axis=0
    )

    applyOrApplyMaps = []
    applyOrApplyMaps.append(
        [{"applymap": c}, {"applymap": c}, {"applymap": c}]
    )  # verbose

    # if no dict, then key would be "applymap"
    applyOrApplyMaps.append([c, c, c])
    applyOrApplyMaps.append([c, {"applymap": c}, c])  # same but partialy

    # if few, then get last given
    applyOrApplyMaps.append(
        [{"applymap": c}, {"applymap": c}]
    )  # 3rd obj has no apply nor applymap
    applyOrApplyMaps.append([{"applymap": c}])  # 2nd and 3nd obj """

    # combine {if few, then get last given} && {no dict}
    applyOrApplyMaps.append([c, c])
    applyOrApplyMaps.append([c])
    applyOrApplyMaps.append(c)  # if just 1 --> no need to be a list

    # default key is "applymap"
    applyOrApplyMaps.append(
        [{"applymap": None}, {"applymap": color_negative_red}, {"apply": applyArgs}]
    )
    applyOrApplyMaps.append([None, color_negative_red, {"apply": applyArgs}])

    df = pd.DataFrame(np.random.randint(0, 100, size=(5, 4)) - 50, columns=list("ABCD"))
    df2 = df.iloc[:2, :2]
    args = ["df2", "df2", "df2"]
    l = locals()

    html_content = ""
    for k in applyOrApplyMaps:
        print(k)
        html_content += "<br>" * 3 + str(k) + "<br>"
        html_content += display_htmlFcn(
            *args,
            _localVars=l,
            _float_prec=0,
            _applyOrApplyMap=k,
            _returnNone=returnNone
            #  _debugBool=1,
        )
    return html_content


def test04(returnNone=False):
    # test float prec
    df = np.eye(3)
    df2 = pd.DataFrame(np.random.rand(5, 4), columns=list("ABCD"))

    args = ["df", "df2", "df2", "df2"]
    l = locals()
    
    floatPrecArr = [1,0,None,3]
    formatValArr = ["{:.0f}","{:.1e}",None,"{:.1f}"]
    formatValArr2 = "{:.1e}"
    floatPrecArr2 = 0

    floatPrecArrZip = [2,4, None,3] # _float_prec just overwrites _formatVal if _formatVal is None
    formatValArrZip = ["{:.0f}",None,None,"{:.1f}"]

    html_content = ""
    if 1:
        html_content += "<br>" * 3 + f"_float_prec: {floatPrecArr}" + "<br>"
        html_content += display_htmlFcn(
            *args,
            _localVars=l,
            _float_prec=floatPrecArr,
            _returnNone=returnNone,
            #  _debugBool=1,
        )
    if 1:
        html_content += "<br>" * 3 + f"_formatVal: {formatValArr}" + "<br>"
        html_content += display_htmlFcn(
            *args,
            _localVars=l,
            _formatVal=formatValArr,
            _returnNone=returnNone,
            #  _debugBool=1,
        )
    if 1:
        html_content += "<br>" * 3 + f"_formatVal: {formatValArrZip}" + "<br>" + f"_float_prec: {floatPrecArrZip}" + "<br>"
        html_content += display_htmlFcn(
            *args,
            _localVars=l,
            _float_prec=floatPrecArrZip,
            _formatVal=formatValArrZip,
            _returnNone=returnNone,
            #  _debugBool=1,
        )
    if 1:
        html_content += "<br>" * 3 + f"_formatVal: {formatValArr2}" + "<br>"
        html_content += display_htmlFcn(
            *args,
            _localVars=l,
            _float_prec=floatPrecArr,
            _formatVal=formatValArr2,
            _returnNone=returnNone,
            #  _debugBool=1,
        )
    if 1:
        html_content += "<br>" * 3 + f"_float_prec: {floatPrecArr2}" + "<br>"
        html_content += display_htmlFcn(
            *args,
            _localVars=l,
            _float_prec=floatPrecArr2,
            _returnNone=returnNone,
            #  _debugBool=1,
        )
    return html_content

def test05():
    df = pd.DataFrame(np.random.randint(0, 100, size=(5, 4)) - 50, columns=list("ABCD"))

    html_content = display_htmlFcn(
        "df",
        "df",
        "df",
        _localVars=locals(),
        _addStr = [None, ".background_gradient(cmap='Greens')", None], 
        _returnNone=False,
        _debugBool=1,
    )
    return html_content

if __name__ == "__main__":
    print("display_htmlFcn!")

    # play:
    #  html_content = test01()
    #  html_content = test02()
    #  html_content = test03()
    #  html_content = test04()
    html_content = test05()

    # testing outside jupyter-notebook
    import webbrowser

    # Save the HTML content to a file
    filename = "display_htmlFcn_00_deleteme.html"
    with open(filename, "w") as file:
        file.write(html_content)

    # Open the file in the Firefox browser
    print(
        "\n\nAuto open HTML file in browser if \033[93m PWD is $HOME,...\033[0m but not VanderPlas directory of current script!!"
    )
    webbrowser.get("firefox").open(filename)

    # open browser from py cmd not working (230710) inside pCloudDrive !!
    #  python3 -c "import webbrowser;webbrowser.register("firefox", None, webbrowser.GenericBrowser("firefox"), 1); webbrowser.open('www.google.com')"
    #  python3 -c "import webbrowser; webbrowser.open('www.google.com')"
    #  sudo python3 -c "import subprocess; subprocess.run(['/snap/bin/firefox', 'https://www.bing.com'])"
    #  python3 display_htmlFcn.py && firefox --new-tab display_htmlFcn_deleteme.html
    # open display_htmlFcn_deleteme.html with double click
