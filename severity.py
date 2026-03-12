def estimate_severity(gravity, ph, osmo, calc):
    score = 0
    if calc > 10: score += 2
    if ph < 5.5: score += 1
    if osmo > 800: score += 2
    if gravity > 1.020: score += 1

    if score <= 2: return "LOW"
    elif score <= 4: return "MEDIUM"
    else: return "HIGH"

def natural_suggestions(severity):
    if severity == "LOW":
        return ["Increase water intake",
                "Reduce salt",
                "Eat citrus fruits"]
    elif severity == "MEDIUM":
        return ["Avoid oxalate foods",
                "Drink lemon water",
                "Moderate protein"]
    else:
        return ["Strict hydration (>3L)",
                "Consult Urologist",
                "Diet control required"]