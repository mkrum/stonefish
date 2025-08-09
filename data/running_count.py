import pandas as pd

df = pd.read_csv(
    "./count_log.csv", header=None, names=["year", "month", "games", "valid", "moves"]
)
running = df[["games", "valid", "moves"]].sum()
print(running)
total = 6956108221

yield_rate = running["moves"] / running["games"]
print(f"Completed: {running['games'] /total * 100.0:.2f}%")
print(f"Yield: {yield_rate * 100.0:.2f}%")
print(f"Current Total (M): {running['moves'] / 10**6:.2f}")
print(f"Estimated Total (M): {(yield_rate * total) / 10**6:.2f}")
