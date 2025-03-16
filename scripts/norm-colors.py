import sys

if __name__ == "__main__":
    color = int(sys.argv[1].replace("#", ""), base=16)
    r, g, b = (color >> 16) & 0xFF, (color >> 8) & 0xFF, color & 0xFF
    r_norm, g_norm, b_norm = float(r) / 255.0, float(g) / 255.0, float(b) / 255.0
    print(f"{r_norm:.2f}, {g_norm:.2f}, {b_norm:.2f}")
