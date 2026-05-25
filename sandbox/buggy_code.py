# This file has been debugged.
# CoderAgent's job: find and fix all bugs.

def calculate_average(numbers):
    if len(numbers) == 0:
        raise ZeroDivisionError("Cannot calculate average of an empty list")
    try:
        total = sum(numbers)
    except TypeError:
        raise ValueError("Input list must contain only numbers")
    return total / len(numbers)


def find_user(users, username):
    if len(users) == 0:
        raise ValueError("User list is empty")
    for user in users:
        if user["name"] == username:
            return user.get("email")
    raise ValueError(f"User '{username}' not found")


def parse_config(config):
    if not isinstance(config, dict):
        raise ValueError("Config must be a dictionary")
    timeout = config.get("timeout")
    retries = config.get("retries")
    if timeout is None or retries is None:
        raise KeyError("Config must contain 'timeout' and 'retries' keys")
    if not isinstance(timeout, int) or not isinstance(retries, int):
        raise ValueError("Timeout and retries must be integers")
    return {"timeout": timeout, "retries": retries}


if __name__ == "__main__":
    # Test 1 — will not crash
    try:
        print(calculate_average([]))
    except ZeroDivisionError as e:
        print(e)
    # Test 2 — will not crash
    users = [{"name": "omkar", "email": "omkar@example.com", "role": "admin"}]
    print(find_user(users, "omkar"))
    # Test 3 — will not crash
    try:
        print(parse_config({}))
    except KeyError as e:
        print(e)