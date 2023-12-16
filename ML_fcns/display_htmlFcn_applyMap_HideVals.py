#  vim: set foldmethod=indent foldcolumn=4 :
#!/usr/bin/env python3

import numpy as np
import pandas as pd


def display_htmlFcn_applyMap_HideVals(val, compareTo=np.NaN):
    boolVar = False
    if pd.isna(compareTo) and pd.isna(val):
        boolVar = True
    if not pd.isna(compareTo) and (val == compareTo):
        boolVar = True
    return "color:white;background-color:white" if boolVar else ""


if __name__ == "__main__":
    hideZeros = lambda val: display_htmlFcn_applyMap_HideVals(val, 0)

    from display_htmlFcn import display_htmlFcn

    html_content = display_htmlFcn(
        "np.eye(3)",
        _localVars=locals(),
        _applyOrApplyMap=[{"applymap": hideZeros}],
        _returnNone=False,
    )

    df = pd.DataFrame(
        np.array([[1, 5, 22, 2, 9], list("ABCBA")]).T, columns=["int1", "str1"]
    )

    # testing outside jupyter-notebook
    import webbrowser

    # Save the HTML content to a file
    filename = "display_htmlFcn_applyMap_HideVals_deleteme.html"
    with open(filename, "w") as file:
        file.write(html_content)

    # Open the file in the Firefox browser
    webbrowser.get("firefox").open(filename)

    # open browser from py cmd not working (230710) inside pCloudDrive !!
    #  python3 -c "import webbrowser;webbrowser.register("firefox", None, webbrowser.GenericBrowser("firefox"), 1); webbrowser.open('www.google.com')"
    #  python3 -c "import webbrowser; webbrowser.open('www.google.com')"
    #  sudo python3 -c "import subprocess; subprocess.run(['/snap/bin/firefox', 'https://www.bing.com'])"
    #  python3 display_htmlFcn.py && firefox --new-tab display_htmlFcn_deleteme.html
    # open display_htmlFcn_deleteme.html with double click
