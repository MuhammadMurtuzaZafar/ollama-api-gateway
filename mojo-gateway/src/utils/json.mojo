"""
JSON parsing and serialization utilities for Mojo.
Lightweight JSON handling without external dependencies.
"""

from collections import Dict, List, Optional


@value
struct JSONValue:
    """
    A JSON value that can hold different types.
    Uses tagged union pattern for type safety.
    """
    var _type: Int  # 0=null, 1=bool, 2=int, 3=float, 4=string, 5=array, 6=object
    var _bool_val: Bool
    var _int_val: Int
    var _float_val: Float64
    var _string_val: String

    @staticmethod
    fn null() -> Self:
        return Self(_type=0, _bool_val=False, _int_val=0, _float_val=0.0, _string_val="")

    @staticmethod
    fn from_bool(val: Bool) -> Self:
        return Self(_type=1, _bool_val=val, _int_val=0, _float_val=0.0, _string_val="")

    @staticmethod
    fn from_int(val: Int) -> Self:
        return Self(_type=2, _bool_val=False, _int_val=val, _float_val=0.0, _string_val="")

    @staticmethod
    fn from_float(val: Float64) -> Self:
        return Self(_type=3, _bool_val=False, _int_val=0, _float_val=val, _string_val="")

    @staticmethod
    fn from_string(val: String) -> Self:
        return Self(_type=4, _bool_val=False, _int_val=0, _float_val=0.0, _string_val=val)

    fn is_null(self) -> Bool:
        return self._type == 0

    fn is_bool(self) -> Bool:
        return self._type == 1

    fn is_int(self) -> Bool:
        return self._type == 2

    fn is_float(self) -> Bool:
        return self._type == 3

    fn is_string(self) -> Bool:
        return self._type == 4

    fn as_bool(self) -> Bool:
        return self._bool_val

    fn as_int(self) -> Int:
        return self._int_val

    fn as_float(self) -> Float64:
        return self._float_val

    fn as_string(self) -> String:
        return self._string_val


struct JSONParser:
    """
    Simple recursive descent JSON parser.
    Parses JSON strings into native Mojo types.
    """
    var _input: String
    var _pos: Int

    fn __init__(out self, input: String):
        self._input = input
        self._pos = 0

    fn parse(mut self) raises -> Dict[String, JSONValue]:
        """Parse a JSON object from the input string."""
        self._skip_whitespace()
        if self._peek() != '{':
            raise Error("Expected '{' at start of JSON object")
        return self._parse_object()

    fn _parse_object(mut self) raises -> Dict[String, JSONValue]:
        """Parse a JSON object { ... }"""
        var result = Dict[String, JSONValue]()
        self._consume('{')
        self._skip_whitespace()

        if self._peek() == '}':
            self._consume('}')
            return result

        while True:
            self._skip_whitespace()
            var key = self._parse_string()
            self._skip_whitespace()
            self._consume(':')
            self._skip_whitespace()
            var value = self._parse_value()
            result[key] = value
            self._skip_whitespace()

            if self._peek() == '}':
                self._consume('}')
                break
            self._consume(',')

        return result

    fn _parse_value(mut self) raises -> JSONValue:
        """Parse any JSON value."""
        self._skip_whitespace()
        var c = self._peek()

        if c == '"':
            return JSONValue.from_string(self._parse_string())
        elif c == 't' or c == 'f':
            return JSONValue.from_bool(self._parse_bool())
        elif c == 'n':
            self._parse_null()
            return JSONValue.null()
        elif c == '-' or (c >= '0' and c <= '9'):
            return self._parse_number()
        else:
            raise Error("Unexpected character in JSON: " + c)

    fn _parse_string(mut self) raises -> String:
        """Parse a JSON string."""
        self._consume('"')
        var result = String("")

        while self._pos < len(self._input):
            var c = self._input[self._pos]
            if c == '"':
                self._pos += 1
                return result
            elif c == '\\':
                self._pos += 1
                if self._pos >= len(self._input):
                    raise Error("Unexpected end of string escape")
                var escaped = self._input[self._pos]
                if escaped == 'n':
                    result += '\n'
                elif escaped == 't':
                    result += '\t'
                elif escaped == 'r':
                    result += '\r'
                elif escaped == '"':
                    result += '"'
                elif escaped == '\\':
                    result += '\\'
                else:
                    result += escaped
            else:
                result += c
            self._pos += 1

        raise Error("Unterminated string")

    fn _parse_number(mut self) raises -> JSONValue:
        """Parse a JSON number."""
        var start = self._pos
        var is_float = False

        if self._peek() == '-':
            self._pos += 1

        while self._pos < len(self._input):
            var c = self._input[self._pos]
            if c >= '0' and c <= '9':
                self._pos += 1
            elif c == '.' or c == 'e' or c == 'E':
                is_float = True
                self._pos += 1
            elif c == '+' or c == '-':
                self._pos += 1
            else:
                break

        var num_str = self._input[start:self._pos]

        if is_float:
            return JSONValue.from_float(atof(num_str))
        else:
            return JSONValue.from_int(atol(num_str))

    fn _parse_bool(mut self) raises -> Bool:
        """Parse a JSON boolean."""
        if self._input[self._pos:self._pos+4] == "true":
            self._pos += 4
            return True
        elif self._input[self._pos:self._pos+5] == "false":
            self._pos += 5
            return False
        else:
            raise Error("Expected 'true' or 'false'")

    fn _parse_null(mut self) raises:
        """Parse a JSON null."""
        if self._input[self._pos:self._pos+4] == "null":
            self._pos += 4
        else:
            raise Error("Expected 'null'")

    fn _peek(self) -> String:
        """Peek at current character."""
        if self._pos >= len(self._input):
            return ""
        return self._input[self._pos]

    fn _consume(mut self, expected: String) raises:
        """Consume an expected character."""
        if self._peek() != expected:
            raise Error("Expected '" + expected + "' but got '" + self._peek() + "'")
        self._pos += 1

    fn _skip_whitespace(mut self):
        """Skip whitespace characters."""
        while self._pos < len(self._input):
            var c = self._input[self._pos]
            if c == ' ' or c == '\t' or c == '\n' or c == '\r':
                self._pos += 1
            else:
                break


fn parse_json(input: String) raises -> Dict[String, JSONValue]:
    """Parse a JSON string into a dictionary."""
    var parser = JSONParser(input)
    return parser.parse()


fn json_get_string(obj: Dict[String, JSONValue], key: String, default: String = "") -> String:
    """Get a string value from a JSON object with a default."""
    if key in obj:
        var val = obj[key]
        if val.is_string():
            return val.as_string()
    return default


fn json_get_int(obj: Dict[String, JSONValue], key: String, default: Int = 0) -> Int:
    """Get an integer value from a JSON object with a default."""
    if key in obj:
        var val = obj[key]
        if val.is_int():
            return val.as_int()
    return default


fn json_get_float(obj: Dict[String, JSONValue], key: String, default: Float64 = 0.0) -> Float64:
    """Get a float value from a JSON object with a default."""
    if key in obj:
        var val = obj[key]
        if val.is_float():
            return val.as_float()
        elif val.is_int():
            return Float64(val.as_int())
    return default


fn json_get_bool(obj: Dict[String, JSONValue], key: String, default: Bool = False) -> Bool:
    """Get a boolean value from a JSON object with a default."""
    if key in obj:
        var val = obj[key]
        if val.is_bool():
            return val.as_bool()
    return default
