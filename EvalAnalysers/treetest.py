class FilterTreeWidget(QtWidgets.QWidget):
    def __init__(self, model_data, parent=None):
        super().__init__(parent)
        self.method_data = []

        # Create QTreeWidget
        self.tree_widget = QtWidgets.QTreeWidget(self)
        self.tree_widget.setHeaderLabel('Method Names')

        # Create top-level items for each model in model_data
        for model_name, method_list in model_data.items():
            model_item = QtWidgets.QTreeWidgetItem(self.tree_widget, [model_name])

            # Create child items for each method in method_list
            for method_name in method_list:
                QtWidgets.QTreeWidgetItem(model_item, [method_name])

        # Set tree widget properties
        self.tree_widget.setSelectionMode(QtWidgets.QAbstractItemView.MultiSelection)
        self.tree_widget.itemSelectionChanged.connect(self.update_method_data)

        # Create layout and add tree widget
        layout = QtWidgets.QVBoxLayout(self)
        layout.addWidget(self.tree_widget)

    def update_method_data(self):
        self.method_data = []
        for item in self.tree_widget.selectedItems():
            method_name = item.text(0)
            model_name = item.parent().text(0)
            self.method_data.append((model_name, method_name))
class MyWindow(QtWidgets.QMainWindow):
    def __init__(self):
        super().__init__()

        # Load UI from file
        uic.loadUi('my_gui.ui', self)

        # Connect buttons to methods
        self.load_button.clicked.connect(self.load_data)
        self.filter_button.clicked.connect(self.filter_data)
        self.analyse_button.clicked.connect(self.analyse_data)

        # Create filter tree widget
        self.filter_tree_widget = FilterTreeWidget(self.model_data)
        self.filter_layout.addWidget(self.filter_tree_widget)

        # Set window properties
        self.show()

    def filter_data(self):
        # Get selected method names from filter tree widget
        method_names = [method[1] for method in self.filter_tree_widget.method_data]

        # Filter current data to include only selected methods
        self.current_data = [line for line in self.current_data if line[2] in method_names]

        # Plot filtered data
        self.plot_data()
