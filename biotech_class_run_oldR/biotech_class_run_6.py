import datetime as dt
import pandas as pd
import math
import numpy as np
import random
from collections import namedtuple
from paul_resources import InformationTable, tprint, rprint
from decorators import my_time_decorator
from Distribution_Module_2 import Distribution, float_to_distribution
from Option_Module import Option, OptionPrice, OptionPriceMC, get_implied_volatility, get_mc_histogram
from Event_Module_2 import Event, SysEvt_PresElection, TakeoutEvent
from functools import reduce
import copy

"""------------------------------Calculations----------------------------------------"""
@my_time_decorator
def mc_simulation(expiry = None, timing_descriptor = None, mc_iterations = 10**5):
    timing_descriptor = 'Q2_2018'

    # Define Events
    event1 = SysEvt_PresElection('CLVS', .02, timing_descriptor)
    event2 = Event('CLVS', .05, 'Q2_2018', 'Q2_Earnings')
    event3 = Event('CLVS', .05, 'Q3_2018', 'Q3_Earnings')
    event4 = Event('CLVS', .075, 'Q3_2018', 'Investor_Day')
    event5 = Event('CLVS', .1, '2H_2018', 'FDA_Approval')
    event6 = TakeoutEvent('CLVS', 1)
    event7 = Event('CLVS', Distribution(pd.read_csv('CLVS.csv')), 'Q2_2018', 'Ph3_Data')
    
    events1 = [event1]
    events2 = [event1, event2]
    events3 = [event1, event2, event3]
    events4 = [event1, event2, event3, event4]
    events5 = [event1, event2, event3, event4, event5]
    events6 = [event1, event2, event3, event4, event5, event6]
    events7 = [event1, event2, event3, event4, event5, event6, event7]
    
    event_groupings = [events1, events2, events3, events4, events5, events6, events7]

    @my_time_decorator
    def get_total_mc_distribution(events, expiry = None, symbol = None, mc_iterations = 10**4):
        """Add the simulation results of individual events to return the total simulated distribution."""
        distributions = map(lambda evt: evt.get_distribution(expiry), events)
        mc_distributions = map(lambda dist: dist.mc_simulation(mc_iterations), distributions)
        return sum(mc_distributions)

    @my_time_decorator
    def get_option_sheet_from_mc_distribution(mc_distribution, expiry = None, strikes = None):
        if strikes is None:
            strikes = np.arange(.5, 1.55, .05)

        call_options = [Option('Call', strike, expiry) for strike in strikes]
        call_prices = list(map(lambda option: OptionPriceMC(option, mc_distribution), call_options))
        call_IVs = list(map(lambda option, option_price: get_implied_volatility(option, option_price), call_options, call_prices))
       
        put_options = [Option('Put', strike, expiry) for strike in strikes]
        put_prices = list(map(lambda option: OptionPriceMC(option, mc_distribution), put_options))
        put_IVs = list(map(lambda option, option_price: get_implied_volatility(option, option_price), put_options, put_prices))

        option_sheet_info = {'Strike': strikes, 'Price': call_prices, 'IV': call_IVs}
        
        option_sheet = pd.DataFrame(option_sheet_info).round(2)
        option_sheet.set_index('Strike', inplace=True)
        option_sheet = option_sheet.loc[:, ['Price', 'IV']]
        print(option_sheet)
        #iterables = [['Price', 'IV'], ['Group 1']]
        #index = pd.MultiIndex.from_product(iterables, names=['Option Info', 'Event Grouping'])
        #option_sheet.rename(columns=index)
        #print(index)
        
        return option_sheet

    @my_time_decorator
    def get_option_sheet_by_event_groupings(event_groupings, expiry):
#        i = 0
#        for grouping in event_groupings:
#            mc_distribution = get_total_mc_distribution(grouping, expiry = expiry)
#            prices = get_option_sheet_from_mc_distribution(mc_distribution, strikes = np.arange(.5, 1.55, .05)).loc[:, ['Price','IV']]
#
#            if event_groupings.index(grouping) == 0:
#                prices_df = prices
#            else:
#                prices_df = pd.merge(prices_df, prices, left_index=True, right_index=True)
#            
#            get_mc_histogram(mc_distribution)
#            
#            i += 1
#        return prices_df
        mc_distributions = list(map(lambda event_grouping: get_total_mc_distribution(event_grouping, expiry), event_groupings))
        [get_mc_histogram(mc_distribution) for mc_distribution in mc_distributions]
        option_sheets = list(map(lambda dist: get_option_sheet_from_mc_distribution(dist, expiry).loc[:, ['IV']], mc_distributions))
        return reduce(lambda x, y: pd.merge(x, y, left_index=True, right_index=True), option_sheets)
        

    event_df = get_option_sheet_by_event_groupings(event_groupings, expiry)
    print(event_df)

timing_descriptor = 'Q2_2018'
expiry = dt.date(2018, 10, 25)
expiry2 = dt.date(2018, 10, 1)

mc_iterations = 10**6
mc_simulation(expiry = expiry, mc_iterations = mc_iterations)
#mc_simulation(expiry = expiry2, mc_iterations = mc_iterations)




def mc_timing_tests():
    event = SystematicEvent('CLVS', .00, 'Ph3_Data')
    distribution = evet.get_distribution()
    pct_moves = distribution.distribution_df.loc[:, 'Pct_Move'].values.tolist()
    weights = distribution.distribution_df.loc[:, 'Prob'].values.tolist()
    
    @my_time_decorator
    def mini_run(k):
        pct_moves = event1.get_distribution().distribution_df.loc[:, 'Pct_Move'].values.tolist()
        weights = evet1.get_distribution().distribution_df.loc[:, 'Prob'].values.tolist()
        results = random.choices(pct_moves, weights=weights, k=k)
        return results1

    results1 = mini_run(10**1)
    results2 = mini_run(10**2)
    results3 = mini_run(10**3)
    results4 = mini_run(10**4)
    results5 = mini_run(10**5)
    results6 = mini_run(10**6)
    results6 = mini_run(10**7)
    #results6 = mini_run(10**8)

@my_time_decorator
def run_takeout_by_expiry():
    event = TakeoutEvent('NBIX', 1)
    option_type = 'Call'
    option_type_2 = 'Put'
    strike = 1.0
    expiries = [dt.date(2018, 4, 20), dt.date(2018, 7, 20), dt.date(2018, 10, 20), dt.date(2019, 1, 20), dt.date(2019, 4, 20)]

    # Preliminary Print Statements
    print("Ann. Takeout Prob: {:.1f}%, Premium: {:.1f}%".format(event.takeout_prob*100, event.takeout_premium*100))

    for expiry in expiries:
        option = Option(option_type, strike, expiry)
        option2 = Option(option_type_2, strike, expiry)

        distribution = event.get_distribution(expiry)

        price = OptionPrice(distribution, option)
        price2 = OptionPrice(distribution, option2)
        
        straddle = price + price2
        print("T.O. by {:%m/%d/%Y}: {:.1f}%".format(expiry, distribution.distribution_df.loc["Takeout", "Prob"]*100), "\n"*0)
        print("Mean Move: {:.1f}%".format(distribution.mean_move*100))
        print("Straddle: {:.1f}%".format(straddle*100), "\n"*1)

@my_time_decorator
def run2():
    expiry = dt.date(2018, 5, 1)
    event1 = TakeoutEvent('CLVS', 1)
    event2 = SysEvt_PresElection('CLVS', .02)
    event3 = SystematicEvent('CLVS', .1, 'Ph3_Data')
    event4 = SystematicEvent('CLVS', .05, 'Investor_Day')
    event5 = SystematicEvent('CLVS', .3, 'FDA Approval')

    distribution1 = event1.get_distribution(expiry)
    distribution2 = event2.get_distribution()
    distribution3 = event3.get_distribution()
    distribution4 = event4.get_distribution()
    distribution5 = event5.get_distribution()

    added_distribution = distribution1 + distribution2 + distribution3 + distribution4 + distribution5
    print(added_distribution)
    print(added_distribution.distribution_df)
    rprint(distribution1.mean_move, distribution2.mean_move, distribution3.mean_move, distribution4.mean_move, distribution5.mean_move, added_distribution.mean_move)

@my_time_decorator
def run3():
    # Define Events
    event1 = TakeoutEvent('CLVS', 1)
    event2 = SysEvt_PresElection('CLVS', .02)
    event3 = SystematicEvent('CLVS', .1, 'Ph3_Data')
    event4 = SystematicEvent('CLVS', .05, 'Investor_Day')
    event5 = SystematicEvent('CLVS', .3, 'FDA_Approval')
    event6 = SystematicEvent('CLVS', .05, 'Q1_Earnings')
    event7 = SystematicEvent('CLVS', .05, 'Q2_Earnings')

    expiry = dt.date(2018, 5, 1)
    events = [event2, event3, event4]
    added_distribution = event1.get_distribution(expiry)
    for event in events:
        added_distribution += event.get_distribution()
    rprint(added_distribution.mean_move)


def run():
    if __name__ == "__main__":
        #-------------------PresElection Setup-----------------#
        PresElectionParams = pd.read_csv("/home/paul/Environments/finance_env/PresElectionParams.csv")
        PresElectionParams.set_index('Stock', inplace=True)

        # Create PresElection Events Dict
        PresElection_Evts = {}
        for stock, move_input in PresElectionParams.itertuples():
            PresElection_Evts[stock] = SysEvt_PresElection(stock, move_input)

        #-------------------Takeout Setup-----------------#
        TakeoutParams = pd.read_csv("TakeoutParams.csv")
        TakeoutParams.set_index('Stock', inplace=True)

        # Create Takeout Events Dict
        Takeout_Evts = {}
        for stock, bucket in TakeoutParams.itertuples():
            Takeout_Evts[stock] = TakeoutEvent(stock, bucket)

        takeout_dict = {}
        for stock, event in Takeout_Evts.items():
            takeout_dict[stock] = (event.takeout_prob, event.takeout_premium)

        takeout_df = pd.DataFrame(takeout_dict).T.round(3)
        takeout_df.rename(columns = {0: 'Prob', 1: 'Premium'}, inplace=True)
        takeout_df.rename_axis('Stock', inplace=True)
        
        
        evt = SystematicEvent('ZFGN', .20)
        evt2 = Event()
        evt3 = SysEvt_PresElection('GM', .05)
        evt4 = TakeoutEvent('NBIX', 1)
        
        print("\n\n\nAll Events---\n", Event.instances, "\n")
        print("Systematic Event---\n", SystematicEvent.instances, "\n")
        print("Presidential Election---\n", SysEvt_PresElection.instances,"\n")
        print("Takeout Event---\n", TakeoutEvent.instances,"\n")
        print(takeout_df.sort_values('Premium', ascending=False))
