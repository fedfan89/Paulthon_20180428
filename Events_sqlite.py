import sqlite3
import pandas as pd
import datetime as dt
from datetime import timedelta
from employee import Employee
from Event_Module import Earnings
import random
from paul_resources import HealthcareSymbols, Symbols, to_pickle
from decorators import my_time_decorator
import pickle

#conn = sqlite3.connect('employee.db')
conn = sqlite3.connect(':memory:')

c = conn.cursor()


c.execute("""CREATE TABLE earnings (
            stock text,
            event_input real,
            timing_descriptor text,
            event_name text
            )""")

def insert_earnings_event(earns):
    with conn:
        c.execute("INSERT INTO earnings VALUES (:stock, :event_input, :timing_descriptor, :event_name)",
                {'stock': earns.stock,
                 'event_input': earns.event_input_value,
                 'timing_descriptor': earns.timing_descriptor.strftime('%Y-%m-%d'),
                 'event_name': earns.event_name})
#@my_time_decorator
def instantiate_earnings_event(params: 'tuple of Earnings params from sqlite db'):
    return Earnings(*params)

@my_time_decorator
def get_earnings_table(symbol=None):
    if symbol is None:
        return pd.read_sql_query("SELECT * FROM earnings", conn)
    else:
        return pd.read_sql_query("SELECT * FROM earnings WHERE stock= ?",
                conn,
                params = (symbol, ))

@my_time_decorator
def get_earnings_events(symbol=None):
    if symbol is None:
        c.execute("SELECT * FROM earnings")
    else:
        c.execute("SELECT * FROM earnings WHERE stock=:stock", {'stock': symbol})
    return [Earnings(*params) for params in c.fetchall()]

@my_time_decorator
def get_earnings_evts_from_pickle(symbol):
    earnings_evts = pickle.load(open('EarningsEvents.pkl', 'rb'))
    return [evt for evt in earnings_evts if evt.stock == symbol]


def get_emps_by_name(lastname):
    c.execute("SELECT * FROM employees WHERE last=:last", {'last': lastname})
    return c.fetchall()

def update_pay(emp, pay):
    with conn:
        c.execute("""UPDATE employees SET pay = :pay
                    WHERE first = :first AND last = :last""",
                    {'first': emp.first, 'last': emp.last, 'pay': pay})
def remove_emp(emp):
    with conn:
        c.execute("DELETE from employees WHERE first = :first AND last = :last",
                    {'first': emp.first, 'last': emp.last})

@my_time_decorator
def create_earnings_events(stocks: 'list of stocks'):
    """Create Earnings Events for a List of Stocks"""
    event_names = ['Q1_2018', 'Q2_2018', 'Q3_2018', 'Q4_2018']
    q1_date_range = list(pd.date_range(dt.date(2018, 1, 1), dt.date(2018, 3, 30)))
    
    earnings_events = []
    for stock in stocks:
        # Set Event Input
        event_input = random.uniform(.03, .07)
        
        # Set Earnings Dates
        q1_date = random.choice(q1_date_range)
        timing_descriptors = [q1_date,
                q1_date + timedelta(90),
                              q1_date + timedelta(180),
                              q1_date + timedelta(270)]

        # Instantiate Earnings Events and append to main list
        for i in range(4):
            earnings_evt = Earnings(stock,
                                    event_input,
                                    timing_descriptors[i],
                                    event_names[i])
            
            earnings_events.append(earnings_evt)
   
    return earnings_events

@my_time_decorator
def insert_events_to_table(earnings_events: 'list of events'):
    for evt in earnings_evts:
        insert_earnings_event(evt)

@my_time_decorator
def get_specific_symbol(symbol, earnings_evts):
    return [evt for evt in earnings_evts if evt.stock == symbol]

@my_time_decorator
def run():
    return [evt for evt in earnings_evts if evt.stock == 'CLVS']

@my_time_decorator
def instantiate(n=1):
    for i in range(n):
        evt = Earnings(*params)

@my_time_decorator
def instantiate_earnings_event_2(params):
    return Earnings(*params)

earnings_evts = pickle.load(open('EarningsEvents.pkl', 'rb'))
insert_events_to_table(earnings_evts)

"""
emps = get_emps_by_name('Doe')
print(emps)

update_pay(emp_2, 95000)
remove_emp(emp_1)

emps = get_emps_by_name('Doe')
print(emps)
"""
#conn.close()



if __name__ == '__main__':
    instantiate(1)
    instantiate(10)
    instantiate(100)
    instantiate(1000)
    #instantiate(10000)

    a = instantiate_earnings_event_2(params)
    print(a.timing_descriptor, type(a.timing_descriptor))
    print('Sup', a)


    #earnings_evts = create_earnings_events(Symbols)
    #to_pickle(earnings_evts, 'EarningsEvents')

    clvs = get_earnings_evts_from_pickle('CLVS')
    print(clvs)

    heya = run()
    print('SUP DIAMOND', heya)

    clvs = get_earnings_events('CLVS')
    print(clvs)

    table = get_earnings_table('CLVS')
    print(table)

    c.execute("SELECT * FROM earnings WHERE stock=:stock", {'stock': 'CLVS'})
    params = c.fetchone()
    print(params)
