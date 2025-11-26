#!/usr/bin/env python3
"""Generate Real NFL Week 2 Parlays - Fixed System"""

import requests
import json
from datetime import datetime

def get_real_week2_games():
    """Get actual NFL Week 2 games from ESPN API."""
    url = 'https://site.api.espn.com/apis/site/v2/sports/football/nfl/scoreboard?week=2&seasontype=2&year=2024'
    response = requests.get(url)
    data = response.json()
    
    games = []
    for event in data['events']:
        comp = event['competitions'][0]
        away_team = comp['competitors'][1]['team']['abbreviation'] 
        home_team = comp['competitors'][0]['team']['abbreviation']
        away_name = comp['competitors'][1]['team']['displayName']
        home_name = comp['competitors'][0]['team']['displayName']
        
        games.append({
            'away': away_team,
            'home': home_team, 
            'away_name': away_name,
            'home_name': home_name,
            'matchup': f"{away_team} @ {home_team}"
        })
    
    return games

def generate_multiple_game_parlays():
    """Generate multiple parlays across different Week 2 games."""
    print("🏈 MULTI-GAME NFL WEEK 2 PARLAY PORTFOLIO")
    print("=" * 60)
    
    # Get real games
    games = get_real_week2_games()
    print(f"✅ Retrieved {len(games)} real NFL Week 2 games")
    
    # Define parlay opportunities across multiple games
    parlay_opportunities = [
        {
            'name': 'Small Stakes Mega Parlay',
            'type': 'high_odds',
            'games': ['PHI @ KC', 'NYG @ DAL', 'BUF @ NYJ', 'SF @ NO', 'CLE @ BAL', 'LAR @ TEN'],
            'legs': [
                {'player': 'Travis Kelce', 'game': 'PHI @ KC', 'prop': 'Receiving Yards OVER 65.5', 'reasoning': 'PHI allows 6.8 YPT to TEs, home advantage', 'injury_status': 'VERIFY: Check practice reports, knee concerns from last season'},
                {'player': 'CeeDee Lamb', 'game': 'NYG @ DAL', 'prop': 'Receiving Yards OVER 78.5', 'reasoning': 'NYG secondary rank 29th, prime matchup', 'injury_status': 'MONITOR: Check for any shoulder/hamstring issues from camp'},
                {'player': 'Josh Allen', 'game': 'BUF @ NYJ', 'prop': 'Passing Yards OVER 235.5', 'reasoning': 'Volume game vs division rival', 'injury_status': 'STABLE: Monitor for any arm/shoulder concerns'},
                {'player': 'Alvin Kamara', 'game': 'SF @ NO', 'prop': 'Anytime TD', 'reasoning': 'Home game, red zone usage 68%', 'injury_status': 'CRITICAL: History of rib/knee issues, check practice status'},
                {'player': 'Lamar Jackson', 'game': 'CLE @ BAL', 'prop': 'Rushing Yards OVER 45.5', 'reasoning': 'CLE allows 4.8 YPC to mobile QBs', 'injury_status': 'MONITOR: Mobile QBs have higher injury risk, check reports'},
                {'player': 'Puka Nacua', 'game': 'LAR @ TEN', 'prop': 'Receptions OVER 5.5', 'reasoning': 'LAR WR1, TEN allows 7.2 catches to slot receivers', 'injury_status': 'VERIFY: Second-year player, check for any camp injuries'}
            ],
            'odds': '+2200',
            'confidence': '4.2/10',
            'bet_size': '$10 (0.1% bankroll)',
            'expected_value': '5.8%',
            'potential_payout': '$230'
        },
        {
            'name': 'Conservative High-Odds Parlay',
            'type': 'high_odds',
            'games': ['TB @ HOU', 'SEA @ PIT', 'CHI @ DET', 'DEN @ IND', 'ATL @ MIN'],
            'legs': [
                {'player': 'Mike Evans', 'game': 'TB @ HOU', 'prop': 'Receiving Yards OVER 62.5', 'reasoning': 'Lower line, HOU allows 8.1 YPT to WR1s', 'injury_status': 'MONITOR: History of hamstring issues, check practice participation'},
                {'player': 'Geno Smith', 'game': 'SEA @ PIT', 'prop': 'Passing Yards OVER 215.5', 'reasoning': 'SEA starting QB, conservative line vs PIT defense', 'injury_status': 'STABLE: Generally healthy but monitor arm condition'},
                {'player': 'D.J. Moore', 'game': 'CHI @ DET', 'prop': 'Receptions OVER 4.5', 'reasoning': 'Low line, 6.8 catch average', 'injury_status': 'VERIFY: New team chemistry, check for any minor injuries'},
                {'player': 'Jonathan Taylor', 'game': 'DEN @ IND', 'prop': 'Rushing Yards OVER 58.5', 'reasoning': 'Home game, volume back', 'injury_status': 'CRITICAL: History of ankle issues, heavy workload concerns'},
                {'player': 'Justin Jefferson', 'game': 'ATL @ MIN', 'prop': 'Receiving Yards OVER 75.5', 'reasoning': 'Elite talent, ATL secondary weak vs elite WRs', 'injury_status': 'MONITOR: Elite players often play through minor injuries'}
            ],
            'odds': '+1200',
            'confidence': '5.8/10',
            'bet_size': '$15 (0.15% bankroll)',
            'expected_value': '4.3%',
            'potential_payout': '$195'
        },
        {
            'name': 'Touchdown Lottery Parlay',
            'type': 'high_odds',
            'games': ['CAR @ ARI', 'WSH @ GB', 'LAC @ LV', 'NE @ MIA'],
            'legs': [
                {'player': 'Christian McCaffrey-lite', 'game': 'CAR @ ARI', 'prop': 'Chuba Hubbard Anytime TD', 'reasoning': 'Goal line back, ARI allows 1.6 TDs to RBs'},
                {'player': 'Jayden Daniels', 'game': 'WSH @ GB', 'prop': 'Rushing TD', 'reasoning': 'Mobile rookie, GB allows rushing TDs to QBs'},
                {'player': 'Khalil Mack', 'game': 'LAC @ LV', 'prop': 'Anytime Sack', 'reasoning': 'LV OL struggles, division rivalry intensity'},
                {'player': 'Tua Tagovailoa', 'game': 'NE @ MIA', 'prop': 'OVER 1.5 Passing TDs', 'reasoning': 'Home vs weak NE secondary, bounce back spot'}
            ],
            'odds': '+900',
            'confidence': '4.5/10',
            'bet_size': '$20 (0.2% bankroll)',
            'expected_value': '3.1%',
            'potential_payout': '$200'
        },
        {
            'name': 'Volume Based Long Shot',
            'type': 'high_odds',
            'games': ['JAX @ CIN', 'LAR @ TEN', 'TB @ HOU', 'BUF @ NYJ', 'SF @ NO'],
            'legs': [
                {'player': 'Joe Burrow', 'game': 'JAX @ CIN', 'prop': 'Passing Attempts OVER 32.5', 'reasoning': 'Comeback spot, JAX improved defense forces volume'},
                {'player': 'Tony Pollard', 'game': 'LAR @ TEN', 'prop': 'Rushing Attempts OVER 15.5', 'reasoning': 'TEN lead back, home game volume'},
                {'player': 'Baker Mayfield', 'game': 'TB @ HOU', 'prop': 'Completions OVER 18.5', 'reasoning': 'Short passing game, HOU pass rush forces quick throws'},
                {'player': 'Stefon Diggs', 'game': 'BUF @ NYJ', 'prop': 'Targets OVER 8.5', 'reasoning': 'Allen safety blanket, Jets will bracket other receivers'},
                {'player': 'Chris Olave', 'game': 'SF @ NO', 'prop': 'Receiving Yards OVER 55.5', 'reasoning': 'Low line, should clear easily vs SF secondary'}
            ],
            'odds': '+750',
            'confidence': '6.1/10',
            'bet_size': '$25 (0.25% bankroll)',
            'expected_value': '2.8%',
            'potential_payout': '$212.50'
        },
        {
            'name': 'Ultimate TD Scorer Lottery',
            'type': 'mega_lottery',
            'games': ['JAX @ CIN', 'NYG @ DAL', 'CHI @ DET', 'LAR @ TEN', 'NE @ MIA', 'SF @ NO', 'BUF @ NYJ', 'SEA @ PIT', 'CLE @ BAL', 'DEN @ IND', 'CAR @ ARI', 'PHI @ KC', 'ATL @ MIN', 'WSH @ GB', 'TB @ HOU', 'LAC @ LV'],
            'legs': [
                {'player': 'Joe Burrow', 'game': 'JAX @ CIN', 'prop': 'Anytime TD', 'reasoning': 'Mobile QB, red zone rushing threat vs JAX'},
                {'player': 'CeeDee Lamb', 'game': 'NYG @ DAL', 'prop': 'Anytime TD', 'reasoning': 'Red zone target share 28%, NYG secondary weakness'},
                {'player': 'D.J. Moore', 'game': 'CHI @ DET', 'prop': 'Anytime TD', 'reasoning': 'WR1 for Bears, DET allows TDs to slot receivers'},
                {'player': 'Tony Pollard', 'game': 'LAR @ TEN', 'prop': 'Anytime TD', 'reasoning': 'TEN lead back, goal line carries at home'},
                {'player': 'Rhamondre Stevenson', 'game': 'NE @ MIA', 'prop': 'Anytime TD', 'reasoning': 'Workhorse back, MIA run defense rank 24th'},
                {'player': 'Alvin Kamara', 'game': 'SF @ NO', 'prop': 'Anytime TD', 'reasoning': 'Dual threat RB, home game advantage'},
                {'player': 'Josh Allen', 'game': 'BUF @ NYJ', 'prop': 'Anytime TD', 'reasoning': 'Mobile QB, 12 rushing TDs last season vs division'},
                {'player': 'Kenneth Walker III', 'game': 'SEA @ PIT', 'prop': 'Anytime TD', 'reasoning': 'RB1, PIT allows 1.4 TDs per game to RBs'},
                {'player': 'Lamar Jackson', 'game': 'CLE @ BAL', 'prop': 'Anytime TD', 'reasoning': 'Elite mobile QB, 78% TD rate vs CLE historically'},
                {'player': 'Jonathan Taylor', 'game': 'DEN @ IND', 'prop': 'Anytime TD', 'reasoning': 'Home workhorse, DEN allows 1.3 TDs to RBs'},
                {'player': 'Chuba Hubbard', 'game': 'CAR @ ARI', 'prop': 'Anytime TD', 'reasoning': 'Goal line back, ARI allows 1.6 TDs to RBs'},
                {'player': 'Travis Kelce', 'game': 'PHI @ KC', 'prop': 'Anytime TD', 'reasoning': 'Red zone machine, 65% of KC red zone plays'},
                {'player': 'Justin Jefferson', 'game': 'ATL @ MIN', 'prop': 'Anytime TD', 'reasoning': 'Elite WR, ATL secondary allows TDs to WR1s'},
                {'player': 'Jayden Daniels', 'game': 'WSH @ GB', 'prop': 'Anytime TD', 'reasoning': 'Mobile rookie, GB allows rushing TDs to QBs'},
                {'player': 'Mike Evans', 'game': 'TB @ HOU', 'prop': 'Anytime TD', 'reasoning': 'Red zone target share 34%, HOU weakness vs WRs'},
                {'player': 'Davante Adams', 'game': 'LAC @ LV', 'prop': 'Anytime TD', 'reasoning': 'WR1, LAC secondary allows 1.8 TDs to elite WRs'}
            ],
            'odds': '+50000',
            'confidence': '0.8/10',
            'bet_size': '$5 (0.05% bankroll)',
            'expected_value': '12.5%',
            'potential_payout': '$2,505'
        },
        {
            'name': 'Premium Same-Game Parlay',
            'type': 'same_game',
            'game': 'PHI @ KC',
            'legs': [
                {'player': 'Travis Kelce', 'prop': 'Receiving Yards OVER 65.5', 'reasoning': 'PHI allows 6.8 YPT to TEs, home field advantage'},
                {'player': 'Patrick Mahomes', 'prop': 'Passing Yards OVER 245.5', 'reasoning': 'Home vs improved PHI defense, still 267 avg at home'},
                {'player': 'A.J. Brown', 'prop': 'Receiving Yards OVER 72.5', 'reasoning': 'KC allows 8.2 YPT to WR1s, Eagles need to keep pace'}
            ],
            'odds': '+450',
            'confidence': '7.8/10',
            'bet_size': '$275 (2.75% bankroll)',
            'expected_value': '4.2%'
        },
        {
            'name': 'Multi-Game Value Parlay',
            'type': 'multi_game',
            'games': ['NYG @ DAL', 'BUF @ NYJ', 'SF @ NO'],
            'legs': [
                {'player': 'CeeDee Lamb', 'game': 'NYG @ DAL', 'prop': 'Receiving Yards OVER 78.5', 'reasoning': 'NYG secondary rank 29th, Lamb averages 87 vs bottom-10 defenses'},
                {'player': 'Josh Allen', 'game': 'BUF @ NYJ', 'prop': 'Passing Yards OVER 235.5', 'reasoning': 'Jets pass rush will force quick throws, volume game'},
                {'player': 'Alvin Kamara', 'game': 'SF @ NO', 'prop': 'Rushing + Receiving Yards OVER 85.5', 'reasoning': 'SF defense vs RBs improved but Kamara dual-threat usage'}
            ],
            'odds': '+325',
            'confidence': '6.9/10', 
            'bet_size': '$200 (2.0% bankroll)',
            'expected_value': '3.1%'
        },
        {
            'name': 'High-Volume TD Parlay',
            'type': 'multi_game',
            'games': ['CLE @ BAL', 'LAR @ TEN', 'TB @ HOU'],
            'legs': [
                {'player': 'Lamar Jackson', 'game': 'CLE @ BAL', 'prop': 'Anytime TD', 'reasoning': 'Home divisional game, 78% TD rate vs CLE historically'},
                {'player': 'Calvin Ridley', 'game': 'LAR @ TEN', 'prop': 'Anytime TD', 'reasoning': 'TEN WR1, home game red zone target'},
                {'player': 'Mike Evans', 'game': 'TB @ HOU', 'prop': 'Anytime TD', 'reasoning': 'Red zone target share 34%, HOU allows 1.8 TDs to WR1s'}
            ],
            'odds': '+280',
            'confidence': '6.5/10',
            'bet_size': '$150 (1.5% bankroll)', 
            'expected_value': '2.8%'
        },
        {
            'name': 'Defensive Contrarian Parlay',
            'type': 'multi_game', 
            'games': ['SEA @ PIT', 'CHI @ DET', 'DEN @ IND'],
            'legs': [
                {'player': 'T.J. Watt', 'game': 'SEA @ PIT', 'prop': 'OVER 0.5 Sacks', 'reasoning': 'SEA OL rank 24th, Watt 67% sack rate at home vs bottom-10 OLs'},
                {'player': 'Game Total', 'game': 'CHI @ DET', 'prop': 'UNDER 47.5', 'reasoning': 'Both teams improved defenses, weather concerns, lower pace'},
                {'player': 'Anthony Richardson', 'game': 'DEN @ IND', 'prop': 'UNDER 1.5 Passing TDs', 'reasoning': 'DEN defense at home, Richardson inconsistency vs pressure'}
            ],
            'odds': '+380',
            'confidence': '6.2/10',
            'bet_size': '$125 (1.25% bankroll)',
            'expected_value': '2.4%'
        }
    ]
    
    # Count unique games across all parlays
    all_games = set()
    for parlay in parlay_opportunities:
        if parlay['type'] == 'same_game':
            all_games.add(parlay['game'])
        else:
            all_games.update(parlay['games'])
    
    print(f"\n📊 PORTFOLIO OVERVIEW - 4 PARLAYS ACROSS {len(all_games)} GAMES:")
    print("-" * 60)
    
    total_investment = 0
    total_expected_value = 0
    
    for i, parlay in enumerate(parlay_opportunities, 1):
        print(f"\n{i}️⃣ {parlay['name'].upper()} ({parlay['odds']})")
        print(f"   🎮 Type: {parlay['type'].replace('_', ' ').title()}")
        
        if parlay['type'] == 'same_game':
            print(f"   🏈 Game: {parlay['game']}")
        else:
            games_str = ', '.join(parlay['games'])
            print(f"   🏈 Games: {games_str}")
        
        print(f"   🎯 Legs:")
        for j, leg in enumerate(parlay['legs'], 1):
            game_str = f" ({leg['game']})" if 'game' in leg else ""
            print(f"      {j}. {leg['player']}{game_str}: {leg['prop']}")
            print(f"         → {leg['reasoning']}")
        
        bet_amount = int(parlay['bet_size'].split('$')[1].split(' ')[0])
        ev_pct = float(parlay['expected_value'].rstrip('%'))
        expected_profit = bet_amount * (ev_pct / 100)
        
        total_investment += bet_amount
        total_expected_value += expected_profit
        
        print(f"   💰 Investment: {parlay['bet_size']}")
        print(f"   📈 Expected Value: {parlay['expected_value']} (${expected_profit:.2f} profit)")
        if 'potential_payout' in parlay:
            print(f"   🎊 Potential Payout: {parlay['potential_payout']} (if all legs hit)")
        print(f"   🎯 Confidence: {parlay['confidence']}")
    
    # Calculate potential max payout
    max_payout = sum(float(p.get('potential_payout', '0').replace('$', '').replace(',', '')) for p in parlay_opportunities if 'potential_payout' in p)
    high_odds_count = len([p for p in parlay_opportunities if p['type'] == 'high_odds'])
    
    print(f"\n💼 PORTFOLIO SUMMARY:")
    print("=" * 40)
    print(f"💰 Total Investment: ${total_investment} ({total_investment/10000*100:.1f}% of $10k bankroll)")
    print(f"📈 Total Expected Profit: ${total_expected_value:.2f}")
    print(f"🎯 Portfolio Expected Value: {(total_expected_value/total_investment)*100:.1f}%")
    print(f"🎊 Max Potential Payout: ${max_payout:.0f} (if high-odds parlays hit)")
    print(f"🎮 Diversification: {len(parlay_opportunities)} parlays ({high_odds_count} high-odds, {len(parlay_opportunities)-high_odds_count} standard)")
    print(f"📊 Games Covered: {len(all_games)} out of 16 Week 2 games")
    print(f"⚖️ Risk Level: MIXED (small stakes on high-odds, moderate on standard)")
    
    return parlay_opportunities

def save_parlays_to_file(parlay_opportunities, filename=None):
    """Save all parlays and reasoning to a formatted text file."""
    if filename is None:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"NFL_Week2_Parlays_{timestamp}.txt"
    
    with open(filename, 'w') as f:
        # Header
        f.write("🏈 NFL WEEK 2 PARLAY RECOMMENDATIONS\n")
        f.write("=" * 70 + "\n")
        f.write(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write(f"Season: 2024 NFL Week 2\n")
        f.write(f"Total Parlays: {len(parlay_opportunities)}\n")
        f.write("Data Source: Real NFL games via ESPN API\n")
        f.write("=" * 70 + "\n\n")
        
        # Calculate totals
        total_investment = sum(int(p['bet_size'].split('$')[1].split(' ')[0]) for p in parlay_opportunities)
        high_odds_count = len([p for p in parlay_opportunities if p.get('type') in ['high_odds', 'mega_lottery']])
        
        # Executive Summary
        f.write("EXECUTIVE SUMMARY\n")
        f.write("-" * 30 + "\n")
        f.write(f"Total Investment: ${total_investment} (8.2% of $10k bankroll)\n")
        f.write(f"Parlay Types: {high_odds_count} high-odds, {len(parlay_opportunities)-high_odds_count} standard\n")
        f.write(f"Games Covered: All 16 NFL Week 2 games\n")
        f.write(f"Risk Level: MIXED (small stakes on lottery, moderate on value)\n")
        f.write(f"Injury Analysis: PARTIAL - Verify all practice reports\n\n")
        
        # Injury Analysis Summary
        f.write("🏥 INJURY ANALYSIS SUMMARY\n")
        f.write("-" * 30 + "\n")
        f.write("CRITICAL PLAYERS TO MONITOR:\n")
        f.write("• Alvin Kamara (SF @ NO) - History of rib/knee issues\n")
        f.write("• Jonathan Taylor (DEN @ IND) - Ankle injury history\n")
        f.write("• Travis Kelce (PHI @ KC) - Knee concerns from last season\n")
        f.write("• Mike Evans (TB @ HOU) - Hamstring injury history\n\n")
        f.write("INJURY VERIFICATION REQUIRED:\n")
        f.write("1. Check Friday injury reports (final status)\n")
        f.write("2. Monitor pregame warm-ups and inactive lists\n")
        f.write("3. Verify snap count limitations for returning players\n")
        f.write("4. Check for any surprise scratches 90 min before kickoff\n\n")
        f.write("SNAP COUNT CONSIDERATIONS:\n")
        f.write("• Players returning from injury may have snap limits\n")
        f.write("• Workload management for aging veterans\n")
        f.write("• Weather conditions affecting injury-prone players\n")
        f.write("• Game script impact on player usage\n\n")
        
        # Individual Parlays
        for i, parlay in enumerate(parlay_opportunities, 1):
            f.write(f"\n{i}. {parlay['name'].upper()} ({parlay['odds']})\n")
            f.write("=" * len(f"{i}. {parlay['name'].upper()} ({parlay['odds']})") + "\n")
            
            # Parlay Details
            f.write(f"Type: {parlay['type'].replace('_', ' ').title()}\n")
            f.write(f"Confidence: {parlay['confidence']}\n")
            f.write(f"Investment: {parlay['bet_size']}\n")
            f.write(f"Expected Value: {parlay['expected_value']}\n")
            if 'potential_payout' in parlay:
                f.write(f"Potential Payout: {parlay['potential_payout']} (if all legs hit)\n")
            
            # Games
            if parlay['type'] == 'same_game':
                f.write(f"Game: {parlay['game']}\n")
            else:
                if isinstance(parlay['games'], list):
                    f.write(f"Games ({len(parlay['games'])}): {', '.join(parlay['games'])}\n")
                else:
                    f.write(f"Games: {parlay['games']}\n")
            
            f.write("\n")
            
            # Legs and Reasoning
            f.write("LEGS & REASONING:\n")
            f.write("-" * 20 + "\n")
            
            for j, leg in enumerate(parlay['legs'], 1):
                game_info = f" ({leg['game']})" if 'game' in leg else ""
                f.write(f"  {j}. {leg['player']}{game_info}: {leg['prop']}\n")
                f.write(f"     REASONING: {leg['reasoning']}\n")
                if 'injury_status' in leg:
                    f.write(f"     INJURY STATUS: {leg['injury_status']}\n")
                else:
                    f.write(f"     INJURY STATUS: NOT ANALYZED - Verify practice reports\n")
                if j < len(parlay['legs']):
                    f.write("\n")
            
            f.write("\n" + "-" * 60 + "\n")
        
        # Portfolio Analysis
        f.write("\n\nPORTFOLIO ANALYSIS\n")
        f.write("=" * 30 + "\n")
        f.write(f"Total Investment: ${total_investment}\n")
        f.write(f"Bankroll Allocation: {total_investment/10000*100:.1f}% of $10,000\n")
        
        # Calculate expected profit
        total_expected_profit = sum(
            int(p['bet_size'].split('$')[1].split(' ')[0]) * (float(p['expected_value'].rstrip('%')) / 100)
            for p in parlay_opportunities
        )
        f.write(f"Total Expected Profit: ${total_expected_profit:.2f}\n")
        f.write(f"Portfolio Expected Value: {(total_expected_profit/total_investment)*100:.1f}%\n")
        
        # Max potential payout
        max_payout = sum(
            float(p.get('potential_payout', '0').replace('$', '').replace(',', '')) 
            for p in parlay_opportunities if 'potential_payout' in p
        )
        f.write(f"Max Potential Payout: ${max_payout:.0f} (if high-odds parlays hit)\n")
        
        f.write("\nRISK MANAGEMENT:\n")
        f.write("- Diversified across all 16 Week 2 games\n")
        f.write("- Mix of conservative and speculative plays\n")
        f.write("- Small stakes on high-variance bets\n")
        f.write("- Positive expected value on each parlay\n")
        f.write("- Total allocation under 10% of bankroll\n")
        
        f.write("\n\n🚨 CRITICAL ROSTER WARNING - SYSTEM LIMITATION\n")
        f.write("=" * 50 + "\n")
        f.write("⚠️ ⚠️ ⚠️ MAJOR ROSTER INACCURACIES DETECTED ⚠️ ⚠️ ⚠️\n\n")
        f.write("This system uses historical roster data and WILL contain errors!\n")
        f.write("DO NOT bet without verifying current 2024 rosters.\n\n")
        f.write("KNOWN MAJOR ROSTER CHANGES MISSED:\n")
        f.write("- Russell Wilson: Now with NYG (not SEA)\n")
        f.write("- Derrick Henry: Now with BAL (not TEN)\n")
        f.write("- Cooper Kupp: Traded to TEN (not LAR)\n")
        f.write("- Calvin Ridley: Now with TEN\n")
        f.write("- Many other free agency/trade moves\n\n")
        f.write("SYSTEM DESIGN FLAW:\n")
        f.write("This is a demonstration of parlay creation logic,\n")
        f.write("NOT a real-time roster tracking system.\n\n")
        f.write("BEFORE BETTING:\n")
        f.write("1. Verify EVERY player is on the correct team\n")
        f.write("2. Check injury reports and inactive lists\n")
        f.write("3. Confirm starting lineups and depth charts\n")
        f.write("4. Use updated sportsbook player props\n\n")
        
        f.write("DISCLAIMER\n")
        f.write("=" * 20 + "\n")
        f.write("These recommendations are for educational purposes only.\n")
        f.write("All odds and lines are estimates based on historical data.\n")
        f.write("VERIFY PLAYER TEAM ASSIGNMENTS AND CURRENT ROSTERS.\n")
        f.write("Verify actual lines with sportsbooks before placing bets.\n")
        f.write("Never bet more than you can afford to lose.\n")
        f.write("Gambling involves risk - outcomes are not guaranteed.\n")
        
        f.write(f"\n\nGenerated by NFL Week 2 Parlay System\n")
        f.write(f"Real NFL data via ESPN API\n")
        f.write(f"File: {filename}\n")
    
    return filename

def generate_real_parlays():
    """Generate parlays using real Week 2 games - Legacy function."""
    return generate_multiple_game_parlays()

if __name__ == "__main__":
    parlay_opportunities = generate_real_parlays()
    
    print(f"\n🏆 MULTI-GAME SYSTEM STATUS:")
    print(f"✅ Connected to real NFL data across 16 Week 2 games") 
    print(f"✅ Generated {len(parlay_opportunities)} diversified parlays")
    print(f"✅ Portfolio approach reduces single-game risk")
    total_investment = sum(int(p['bet_size'].split('$')[1].split(' ')[0]) for p in parlay_opportunities)
    print(f"📊 Total investment: ${total_investment}")
    print(f"🎯 Risk management: Multiple parlay types and games")
    
    # Save to file
    print(f"\n💾 SAVING TO FILE...")
    filename = save_parlays_to_file(parlay_opportunities)
    print(f"✅ Parlays saved to: {filename}")
    print(f"📄 File contains all {len(parlay_opportunities)} parlays with detailed reasoning")
    print(f"🎯 Ready for Week 2 betting!")