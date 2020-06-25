 <?php
 // Check if the form is submitted
if ( isset( $_POST['submit'] ) ) {

// retrieve the form data by using the element's name attributes value as key


$currencyA = $_POST['currencyA'];
$currencyB = $_POST['currencyB'];
$TimePeriod = $_POST['TimePeriod'];

// display the results
#echo 'currencyA is ' . $currencyA;
#echo 'currencyB is ' . $currencyB;
shell_exec("sudo touch info.conf");
$myfile = fopen("/var/www/html/forex_analysis/conf/info.conf", "w") or die("Unable to open file!");
fwrite($myfile,"[info]
CURRENCY_X=$currencyA
CURRENCY_Y=$currencyB
INTERVAL=2
FORECAST=$TimePeriod

[major]
FILES = cpi, exports, imports, gdp, population
USD = United States
EUR = Germany
INR = India
GBP = United Kingdom
CAD = Canada
CNY = China
IDR = Indonesia
MYR = Malaysia
SGD = Singapore
AUD = Australia
");

fclose($myfile);
$res = chdir("/var/www/html/forex_analysis/");
$res = shell_exec("./starter.sh");
//$res = exec
#	echo "\n"."done"."\r\n".$res;
header('Location: result.html');
exit;

}
?>
