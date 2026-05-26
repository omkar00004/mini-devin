# sandbox/multi_bug.py
# Three independent bug clusters — one per function

def calculate_stats(numbers):
    """Bug: crashes on empty list (ZeroDivisionError + min/max)"""
    if len(numbers) == 0:
        return {
            "average": 0,
            "min":     0,
            "max":     0,
            "count":   0
        }
    total   = sum(numbers)
    average = total / len(numbers)
    return {
        "average": average,
        "min":     min(numbers),
        "max":     max(numbers),
        "count":   len(numbers)
    }


def load_user_data(user_dict):
    """Bug: KeyError if email or role missing"""
    return {
        "name":  user_dict.get("name", ""),
        "email": user_dict.get("email", ""),
        "role":  user_dict.get("role", "")
    }


def process_config(config):
    """Bug: KeyError on optional keys"""
    return {
        "host":    config.get("host", ""),
        "port":    config.get("port", 0),
        "timeout": config.get("timeout", 0),
        "retries": config.get("retries", 0),
        "debug":   config.get("debug", False)
    }


if __name__ == "__main__":
    print(calculate_stats([10, 20, 30]))
    print(load_user_data({"name": "omkar", "email": "o@o.com", "role": "admin"}))
    print(process_config({"host": "localhost", "port": 8080,
                          "timeout": 30, "retries": 3, "debug": False}))