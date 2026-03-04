def fibonacci(n):
    """
    Generate the nth Fibonacci number (0-indexed).
    
    Args:
        n: The position in the Fibonacci sequence
        
    Returns:
        The nth Fibonacci number
    """
    if n <= 0:
        return 0
    elif n == 1:
        return 1
    else:
        a, b = 0, 1
        for _ in range(2, n + 1):
            a, b = b, a + b
        return b


def fibonacci_sequence(n):
    """
    Generate the first n Fibonacci numbers.
    
    Args:
        n: How many Fibonacci numbers to generate
        
    Returns:
        A list of the first n Fibonacci numbers
    """
    if n <= 0:
        return []
    elif n == 1:
        return [0]
    
    sequence = [0, 1]
    for i in range(2, n):
        sequence.append(sequence[i-1] + sequence[i-2])
    return sequence


def fibonacci_recursive(n):
    """
    Generate the nth Fibonacci number using recursion (slower, for educational purposes).
    
    Args:
        n: The position in the Fibonacci sequence
        
    Returns:
        The nth Fibonacci number
    """
    if n <= 0:
        return 0
    elif n == 1:
        return 1
    else:
        return fibonacci_recursive(n - 1) + fibonacci_recursive(n - 2)


# Example usage
if __name__ == "__main__":
    # Get the 10th Fibonacci number
    print(f"10th Fibonacci number: {fibonacci(10)}")
    
    # Get the first 10 Fibonacci numbers
    print(f"First 10 Fibonacci numbers: {fibonacci_sequence(10)}")
    
    # Using recursive version (slower)
    print(f"10th Fibonacci (recursive): {fibonacci_recursive(10)}")
