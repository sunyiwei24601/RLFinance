# -*- coding: utf-8 -*-
"""
    :author: Grey Li (李辉)
    :url: http://greyli.com
    :copyright: © 2018 Grey Li <withlihui@gmail.com>
    :license: MIT, see LICENSE for more details.
"""
from flask import render_template, Blueprint, current_app, make_response, jsonify, g, request
from flask_babel import _
from flask_login import current_user

from todoism.extensions import db
from todoism.models import StockPrice, User, Item
import random

home_bp = Blueprint('home', __name__)


@home_bp.route('/')
def index():
    return render_template('index.html')


@home_bp.route('/intro')
def intro():
    return render_template('_intro.html')


@home_bp.route('/set-locale/<locale>')
def set_locale(locale):
    if locale not in current_app.config['TODOISM_LOCALES']:
        return jsonify(message=_('Invalid locale.')), 404

    response = make_response(jsonify(message=_('Setting updated.')))
    if current_user.is_authenticated:
        current_user.locale = locale
        db.session.commit()
    else:
        response.set_cookie('locale', locale, max_age=60 * 60 * 24 * 30)
    return response


@home_bp.route('/portfolio')
def portfolio():
    return render_template('_portfolio.html')


@home_bp.route('/data')
def get_data():
    return jsonify(message=[12, 19, 3, 5, 2, 3])


@home_bp.route('/get_portfolio')
def get_portfolio():
    money = request.args.get('money')
    date = request.args.get('date')
    stock_type = request.args.get('tp')
    print(money, date, stock_type)
    items = Item.query.filter_by(author_id=current_user.id)
    stocks = [item.body for item in items]
    print(stocks)
    result = predict(10000, stocks, date='2022-01-29')
    return jsonify(labels=list(result.keys()), data=list(result.values()))


def predict(money, stocks, date):
    return {
        stock: random.randint(-100, 100)
        for stock in stocks
    }
