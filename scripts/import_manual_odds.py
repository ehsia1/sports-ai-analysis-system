#!/usr/bin/env python3
"""
Import manually entered odds from CSV.
"""
import sys
from pathlib import Path
import argparse

# Add src directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.sports_betting.data.manual_odds_entry import ManualOddsImporter


def main():
    parser = argparse.ArgumentParser(
        description='Import manually entered odds from CSV'
    )
    parser.add_argument(
        'csv_file',
        nargs='?',
        help='Path to CSV file with odds'
    )
    parser.add_argument(
        '--create-template',
        action='store_true',
        help='Create a CSV template for manual entry'
    )

    args = parser.parse_args()

    importer = ManualOddsImporter()

    if args.create_template:
        template_path = importer.create_template()
        print(f"✓ Template created at: {template_path}")
        print("\nInstructions:")
        print("1. Open the template in Excel or text editor")
        print("2. Fill in player names, odds, and lines from sportsbook")
        print("3. Save as new file (e.g., odds_2024-12-01.csv)")
        print("4. Import: python scripts/import_manual_odds.py <your_file>.csv")
        return

    if not args.csv_file:
        print("Error: Please provide a CSV file to import")
        print("\nUsage:")
        print("  Create template: python scripts/import_manual_odds.py --create-template")
        print("  Import odds: python scripts/import_manual_odds.py <csv_file>")
        sys.exit(1)

    csv_path = Path(args.csv_file)

    if not csv_path.exists():
        print(f"❌ File not found: {csv_path}")
        sys.exit(1)

    print("=" * 60)
    print("IMPORTING MANUAL ODDS")
    print("=" * 60)
    print(f"Source: {csv_path}")
    print()

    imported = importer.import_from_csv(csv_path)

    if imported > 0:
        print(f"\n✓ Successfully imported {imported} props")
        print("\nNext steps:")
        print("  1. Generate predictions for these players")
        print("  2. Calculate edges: python scripts/calculate_edges.py")
        print("  3. Review betting opportunities")
    else:
        print("\n❌ No props imported")
        print("Check:")
        print("  - Player names match database exactly")
        print("  - Games exist for specified dates")
        print("  - CSV format is correct")


if __name__ == "__main__":
    main()
