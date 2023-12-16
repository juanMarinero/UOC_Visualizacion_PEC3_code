#  vim: set foldmethod=indent foldcolumn=4 :
#!/usr/bin/env python3

import pandas as pd
import matplotlib.pyplot as plt
import mpld3
from IPython.display import display_html
from bs4 import BeautifulSoup
import inspect


def html2file(html_content, filename = "display_htmlFcn_00_deleteme.html", mode="w"):
    
    # Save the HTML content to a file
    with open(filename, mode) as file:
        file.write(html_content)

def open_html(filename = "display_htmlFcn_00_deleteme.html"):
    
    # Open the file in the Firefox browser
    print(
        "\n\nAuto open HTML file in browser if \033[93m PWD is $HOME,...\033[0m but not VanderPlas directory of current script!!"
    )

    # testing outside jupyter-notebook
    import webbrowser
    webbrowser.get("firefox").open(filename)
    
    # open browser from py cmd not working (230710) inside pCloudDrive !!
    #  python3 -c "import webbrowser;webbrowser.register("firefox", None, webbrowser.GenericBrowser("firefox"), 1); webbrowser.open('www.google.com')"
    #  python3 -c "import webbrowser; webbrowser.open('www.google.com')"
    #  sudo python3 -c "import subprocess; subprocess.run(['/snap/bin/firefox', 'https://www.bing.com'])"
    #  python3 display_htmlFcn.py && firefox --new-tab display_htmlFcn_deleteme.html
    # open display_htmlFcn_deleteme.html with double click

def getFig():
    iris_df = pd.read_csv('https://raw.githubusercontent.com/mwaskom/seaborn-data/master/iris.csv')

    plt.ioff() # prevent plots from being displayed in the output of Jupyter Notebook

    fig, ax = plt.subplots()
    for species, group in iris_df.groupby('species'):
        ax.scatter(group['sepal_length'], group['sepal_width'], label=species)
    ax.set_xlabel('Sepal Length')
    ax.set_ylabel('Sepal Width')
    ax.legend() 

    plt.ion() # revert plt.ioff()

    return fig

def getFigSubplot():
    import numpy as np
    x = np.array(range(100))
    y = np.sin(x)

    plt.ioff() # prevent plots from being displayed in the output of Jupyter Notebook
    fig, ax = plt.subplots(2, 2, sharex='col', sharey='row')

    ax[0,0].scatter(x,y)
    ax[0,1].plot(x,y) # or plt.plot(x,y,ax=ax[0,1])
    ax[1,0].hist(abs(y)*1e2, bins=10, density=True, histtype='bar', rwidth=0.8)
    ax[1,1].bar(x,abs(y)/25)
    plt.ion() # revert plt.ioff()

    return fig

def get_html_df(caption="iris_df.groupby('species').mean()"):
    iris_df = pd.read_csv('https://raw.githubusercontent.com/mwaskom/seaborn-data/master/iris.csv')
    html_df = iris_df.groupby('species')[['sepal_length','sepal_width']].mean()\
            .style.set_table_attributes("style='display:inline'")\
            .set_caption(caption)._repr_html_()
    return html_df

def display_htmlFcn_plots(fig,
                          term = 3, # 1, 2 or 3,
                          head="", file="deleteme.jpg", caption="",
                          width=300.0, height=300.0
                          ):

    html_plot = main_mpld3(fig, width=width, height=height) 

    frame = inspect.currentframe()
    args, _, _, values = inspect.getargvalues(frame)
    kwargs = {arg: values[arg] for arg in args}
    [kwargs.pop(key, None) for key in ["fig", "term"]]

    match term:
        case 1:
            mpld3.enable_notebook()
            # html_plot not modified
        case 2:
            html_plot = fig2file2html(plt=plt, **kwargs)
        case 3:
            html_plot = fig2file2html(html_plot=html_plot, **kwargs)
    plt.close()
    return html_plot

def main_mpld3(fig, width, height):
    if 0:
        import uuid
        figid = str(uuid.uuid4())
        print(f"\t~~~ {'figid':20s}: {figid}")
        html_plot = mpld3.fig_to_html(fig, figid=figid)
    else:
        html_plot = mpld3.fig_to_html(fig)
    # print(html_plot)

    html_plot = editHTML(html_plot, width, height)
    # print(html_plot)
    return html_plot

def editHTML(html_plot, width, height):
    soup = BeautifulSoup(html_plot, 'html.parser')
    aux = soup.prettify()

    toMatch = '"width": 640.0, "height": 480.0'
    toReplace = f'"width": {width}, "height": {height}' # shows plot, but not inline
    # toReplace = f'"width": {width}, "height": {height}, "style"="display:inline;"' # NOT shows plot
    # toReplace = f'"width": {width}, "height": {height}, "display"="inline"' # NOT shows plot
    modified_html = aux.replace(toMatch, toReplace)

    # no effect to inline
    toMatch = '"drawstyle": "default"'
    toReplace = '"drawstyle": "inline"'
    modified_html = modified_html.replace(toMatch, toReplace)

    # no effect to inline
    toMatch = '<style>\n</style>'
    toReplace = ''
    modified_html = modified_html.replace(toMatch, toReplace)

    soup = BeautifulSoup(modified_html, 'html.parser')
    aux = soup.prettify()

    return aux

def fig2file2html(plt=None, html_plot=None, head="", file="deleteme.jpg", caption="",
                  width=300.0, height=300.0):
    if (plt is None) and (html_plot is None):
        return Error
    if plt is not None:
        plt.savefig(file)
        html_img = f'<img src={file} alt="" border=3 height={height} width={width}></img>'
    if html_plot is not None:
        hr = 4*"&nbsp;"
        html_img = hr + html_plot + hr

    # <img> --> no inline
    html_plot= html_img.replace("<img", "<img style='display:inline ")

    # <div> --> no inline
    html_plot= f"""<div style='display:inline'>
    {html_img}
    </div>
    """

    # <table> --> YES inline
    html_plot= f"""<table style='display:inline'>
    <caption>{caption}</caption>
    <tr><th>{head}</th><tr>
    <tr><td>
    {html_img}
    </td></tr>
    </table>
    """
    return html_plot


def test01():
    if 0:
        fig = getFig()
    else:
        fig = getFigSubplot()
    html_df = get_html_df()
    html_plot = display_htmlFcn_plots(fig)

    print("2 dfs inline:")
    display_html(html_df + html_df, raw=True) # YES success!

    print("df and plot inline:")
    html = html_df + html_plot
    display_html(html, raw=True) # inline if term=3 or term=2

    return html

def test02():
    fig = getFig()
    html_df = get_html_df(caption="")
    html_plot = display_htmlFcn_plots(fig,
                                      term = 3, # 1, 2 or 3
                                      head="Plot", file="deleteme.jpg", caption="HTML-repr's caption tag",
                                      width=650.0, height=650.0,
                                      )
    html = html_df + html_plot
    display_html(html, raw=True)
    return html

def test03(preventUndesiredEffect=True):
    fig1 = getFig()
    fig2 = getFigSubplot()
    html_df = get_html_df()
    html_plot1 = display_htmlFcn_plots(fig1)
    html_plot2 = display_htmlFcn_plots(fig2)

    html = html_df + html_plot1
    display_html(html, raw=True) # inline if term=3 or term=2
    html2file(html)

    if preventUndesiredEffect:
        html_plot1_new = display_htmlFcn_plots(fig1)
        html = html_df + html_plot1_new
    html += html_plot2
    display_html(html, raw=True) # inline if term=3 or term=2
    return html



if __name__ == "__main__":
    print("display_htmlFcn_plots!")

    # play
    mode = "w"
    #  html_content = test01()
    #  html_content = test02()

    html_content = test03(); mode="a" # desired effect
    #  html_content = test03(preventUndesiredEffect=False); mode="a" 



    # for all
    html2file(html_content, mode=mode)
    open_html()
