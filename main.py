import sys
from PyQt5.QtWidgets import QApplication
from GUI_test import Work


if __name__ == '__main__':
    app = QApplication(sys.argv)
    work = Work()
    work.show()
    sys.exit(app.exec_())