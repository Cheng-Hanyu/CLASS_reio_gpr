import sys
print("Python version information:")
print(sys.version)
print(sys.version_info)

# Test f-string
name = "World"
print(f"Hello, {name}!")

# Test for Python 3.6+ features
print("Testing more Python 3.6+ features:")
test_list = [1, 2, 3, 4, 5]
print(f"List: {test_list}")

# Test class variables type annotations (3.6+)
class Test:
    x: int = 10
    
test = Test()
print(f"Test.x = {test.x}")

print("If you see this, all tests passed!")
