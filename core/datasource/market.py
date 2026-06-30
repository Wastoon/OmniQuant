class MarketResolver:

    @staticmethod
    def detect(code):

        if code.isdigit():

            if code.startswith(("6", "5")):
                return "SH"

            if code.startswith(("0", "3")):
                return "SZ"

            if code.startswith("688"):
                return "STAR"

        if code.isdigit() and len(code) == 5:
            return "HK"

        return "US"

    @staticmethod
    def secid(code):

        m = MarketResolver.detect(code)

        if m == "SH":
            return f"1.{code}"

        if m == "SZ":
            return f"0.{code}"

        raise ValueError("Only A-share supported in EM API")
