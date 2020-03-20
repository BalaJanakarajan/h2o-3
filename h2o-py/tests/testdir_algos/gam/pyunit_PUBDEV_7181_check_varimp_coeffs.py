from __future__ import division
from __future__ import print_function
from past.utils import old_div
import sys
sys.path.insert(1, "../../../")
import h2o
from tests import pyunit_utils
from h2o.estimators.gam import H2OGeneralizedAdditiveEstimator

# In this test, we check and make sure that we can get the various model coefficients and variable importance
def test_gam_coeffs_varimp():
    print("Checking coefficients and variable importance for binomial")
    h2o_data = h2o.import_file(
        path=pyunit_utils.locate("smalldata/glm_test/binomial_20_cols_10KRows.csv"))
    h2o_data["C1"] = h2o_data["C1"].asfactor()
    h2o_data["C2"] = h2o_data["C2"].asfactor()
    myX = ["C1", "C2"]
    myY = "C21"
    h2o_data["C21"] = h2o_data["C21"].asfactor()
    buildModelCoeffVarimpCheck(h2o_data, myX, myY, ["C11", "C12", "C13"], 'binomial')

    print("Checking coefficients and variable importance for gaussian")
    h2o_data = h2o.import_file(
    path=pyunit_utils.locate("smalldata/glm_test/gaussian_20cols_10000Rows.csv"))
    h2o_data["C1"] = h2o_data["C1"].asfactor()
    h2o_data["C2"] = h2o_data["C2"].asfactor()
    myX = ["C1", "C2"]
    myY = "C21"
    h2o_data["C21"] = h2o_data["C21"].asfactor()
    buildModelCoeffVarimpCheck(h2o_data, myX, myY, ["C11", "C12", "C13"], 'gaussian')

    print("Checking coefficients and variable importance for gaussian")
    h2o_data = h2o.import_file(
    path=pyunit_utils.locate("smalldata/glm_test/multinomial_10_classes_10_cols_10000_Rows_train.csv"))
    h2o_data["C1"] = h2o_data["C1"].asfactor()
    h2o_data["C2"] = h2o_data["C2"].asfactor()
    myX = ["C1", "C2"]
    myY = "C11"
    h2o_data["C11"] = h2o_data["C11"].asfactor()
    buildModelCoeffVarimpCheck(h2o_data, myX, myY, ["C6", "C7", "C8"], 'gaussian')
    
    print("gam coeff/varimp test completed successfully")    
    
def buildModelCoeffVarimpCheck(train_data, x, y, gamX, family):
    numKnots = [5,6,7]
    h2o_model = H2OGeneralizedAdditiveEstimator(family=family, gam_X=gamX,  scale = [1,1,1], k=numKnots)
    h2o_model.train(x=x, y=y, training_frame=train_data)
    h2oCoeffs = h2o_model.coef()
    h2oCoeffsStandardized = h2o_model.coef_norm()
    varimp = h2o_model.varimp()

if __name__ == "__main__":
    h2o.init(ip="192.168.86.39", port=54321, strict_version_check=False)
    pyunit_utils.standalone_test(test_gam_coeffs_varimp)
else:
    h2o.init(ip="192.168.86.39", port=54321, strict_version_check=False)
    test_gam_coeffs_varimp()
