# -*- coding:utf-8 -*-

from flask import Flask, render_template, request
import sys
import os
sys.path.append(os.path.abspath('./src'))
from sentence_embd.auto_summary import AutoSummary
import logging
import gc


LOG_FORMAT = '%(asctime)s - %(levelname)s - %(message)s'
logging.basicConfig(level=logging.INFO, filename='./log/summary.log', format=LOG_FORMAT)
app = Flask(__name__)
auto_summary = AutoSummary()
auto_summary.prepare_data()
logging.info('auto summary finished preparing')


@app.route('/get_summary', methods=['GET', 'POST'], strict_slashes=False)
def get_summary():
    logging.info('get request')
    news = request.args.get('newsContent')
    title = request.args.get('title')
    logging.info('news : {}'.format(news))
    logging.info('title : {}'.format(title))
    summary = ''
    try:
        if not news:
            summary = ''
        elif len(title.strip()) == 0:
            summary = auto_summary.summary(news)
        else:
            summary = auto_summary.summary_with_title(news, title)
    except Exception as e:
        logging.info('Exception : {}'.format(str(e)))
    logging.info('summary : {}'.format(summary))
    return render_template('index.html', summary=summary)


if __name__ == '__main__':
    app.run(host='127.0.0.1', port=9999)
