
/*=================================================*/
/*=================================================*/
/*THIS IS WHERE FILES WILL BE SAVED - CHANGE TO YOUR OWN SCRATCH DIRECTORY WHICH YOU HAVE ACCES TOO*/
libname taq_20 '/scratch/hecca/saad'; 
/*e.g. libname taq_20  '/scratch/hecca/TradingClub'; */
/*=================================================*/



/* STEP 1: RETRIEVE DAILY TRADE AND QUOTE (DTAQ) FILES */
    libname nbbomsec '/wrds/nyse/sasdata/taqms/nbbo';
    libname cqmsec '/wrds/nyse/sasdata/taqms/cq';
    libname ctmsec '/wrds/nyse/sasdata/taqms/ct';


%let outdir=%sysfunc(getoption(work));
options dlcreatedir;
libname out20 "&outdir./my_test_subdir_20";
*options nosource nonotes errors=0;
OPTIONS SOURCE MACROGEN SYMBOLGEN MPRINT SORTSIZE=MAX COMPRESS=YES;






%macro NonZero(ds);
/*first check if data exist, if it exist than count obs*/

%global N_OBS;
 %if %sysfunc(exist(&ds)) %then %do;
  %let DSID=%sysfunc(OPEN(&ds.,IN));
    %let N_OBS=%sysfunc(ATTRN(&DSID,NOBS));
    %let RC=%sysfunc(CLOSE(&DSID));
    
%put &N_OBS;


	
 %end;
%mend NonZero;

/*
s_orderflow = size based orderflow
v_orderflow= volume based orderflow
orderflow = trade direction based orderflow
ric= symbol
voume = dollar volume traded
NBB= last best bid prevailing within the time interval
NBO= last best offer price prevailiing with the given time interval
You can compute dollar quoted spread as NBO-NBB
Midpoint price is (NBBO+NBB)/2
percent quoted spread relative to midpoint is ((NBO-NBB)/MIDPOINT)*100
Price Impact of Trade: Use future price movements in the direction of current trade:
Suppose orderflow>0 and e.g. 30 second into future midpoint increases


Price Impact: regression analysis midpoint returns on orderflow
Non parametric Price Impact:

orderflow*(midpoint return now versus say 30 seconds)
For price impact:
Suppose order flow is "9:35:01" implies buys minus sells from "9:35:00" to "9:35:01" buys minus sells
For price impact you want to look at midpoint move right before the order-imblanace up untill the future
Return: is going to be "9:35:00" to "9:35:30"

Fit a bivariate VAR
(returns, orderflow)

To test for borwnian look up variance ratio: simple non parametric method

You can use all sorts of methods for forecasting using relationship between returns, past returns, orderflow and past orderflow*/



data taq_20.TAQ_SP_500_2020_1sec;

format ric $char8.;
format date_time datetime23.3;
format NBB best12.;
format NBO best12.;
format price best12.;
format volume best12.;
format s_orderflow best12.;
format v_orderflow best12.;
format orderflow best12.;
format trades best12.;
format quotes best12.;
format time_m2 time20.6;
format open_price $5.;


run;








%macro SCANLOOP_0(VAR1,VAR2,VAR3,VAR4);



%let sym=&var1;
%let nbbo_file=&var2;
%let cqm_file=&var3;
%let ctm_file=&var4;





    /* Retrieve NBBO data */
    data out20.NBBO;

        /* Enter NBBO file names in YYYYMMDD format for the dates you want */
        set &nbbo_file;

		/* Enter company tickers you want */
        where sym_root in ("&sym") and SYM_SUFFIX=' ' and

        /* Quotes are retrieved prior to market open time to ensure NBBO 
		   Quotes are available for beginning of the day trades */
        (("9:00:00.000000000"t) <= time_m <= ("16:00:00.000000000"t));
        format date date9.;
        format time_m part_time trf_time TIME20.9;
    run;

    /* Retrieve Quote data */
    data out20.quoteAB;

        /* Enter Quote file names in YYYYMMDD format for the same dates */
        set &cqm_file;

		/* Enter the same company tickers as above */
        where sym_root in ("&sym") and SYM_SUFFIX=' ' and
        /* Quotes are retrieved prior to market open time to ensure NBBO 
		   Quotes are available for beginning of the day trades*/
        (("9:00:00.000000000"t) <= time_m <= ("16:00:00.000000000"t));
        format date date9.;
        format time_m part_time trf_time TIME20.9;
		
    run;

    /* Retrieve Trade data=== */
    data out20.trade;

        /* Enter Trade file names in YYYYMMDD format for the same dates */
        set &ctm_file;

		/* Enter the same company tickers as above */
        where sym_root in ("&sym") and SYM_SUFFIX=' ' and

        /* Retrieve trades during normal market hours */
        (("9:30:00.000000000"t) <= time_m <= ("16:00:00.000000000"t));
        type='T';
        format date date9.;
        format time_m part_time trf_time TIME20.9;

    run;


	/* STEP 2: CLEAN THE DTAQ NBBO FILE */ 

data out20.NBBO;
    set out20.NBBO;

    /* Quote Condition must be normal (i.e., A,B,H,O,R,W) */
    if Qu_Cond not in ('A','B','H','O','R','W') then delete;

	/* If canceled then delete */
    if Qu_Cancel='B' then delete;

	/* if both ask and bid are set to 0 or . then delete */
    if Best_Ask le 0 and Best_Bid le 0 then delete;
    if Best_Asksiz le 0 and Best_Bidsiz le 0 then delete;
    if Best_Ask = . and Best_Bid = . then delete;
    if Best_Asksiz = . and Best_Bidsiz = . then delete;

	/* Create spread and midpoint */
    Spread=Best_Ask-Best_Bid;
    Midpoint=(Best_Ask+Best_Bid)/2;

	/* If size/price = 0 or . then price/size is set to . */
    if Best_Ask le 0 then do;
        Best_Ask=.;
        Best_Asksiz=.;
    end;
    if Best_Ask=. then Best_Asksiz=.;
    if Best_Asksiz le 0 then do;
        Best_Ask=.;
        Best_Asksiz=.;
    end;
    if Best_Asksiz=. then Best_Ask=.;
    if Best_Bid le 0 then do;
        Best_Bid=.;
        Best_Bidsiz=.;
    end;
    if Best_Bid=. then Best_Bidsiz=.;
    if Best_Bidsiz le 0 then do;
        Best_Bid=.;
        Best_Bidsiz=.;
    end;
    if Best_Bidsiz=. then Best_Bid=.;

	/*	Bid/Ask size are in round lots, replace with new shares variable*/
	Best_BidSizeShares = Best_BidSiz * 100;
	Best_AskSizeShares = Best_AskSiz * 100;
run;

/* STEP 3: GET PREVIOUS MIDPOINT */

proc sort 
    data=out20.NBBO (drop = Best_BidSiz Best_AskSiz);
    by sym_root date;
run; 

data out20.NBBO;
    set out20.NBBO;
    by sym_root date;
    lmid=lag(Midpoint);
    if first.sym_root or first.date then lmid=.;
    lm25=lmid-2.5;
    lp25=lmid+2.5;
run;

/* If the quoted spread is greater than $5.00 and the bid (ask) price is less
   (greater) than the previous midpoint - $2.50 (previous midpoint + $2.50), 
   then the bid (ask) is not considered. */

data out20.NBBO;
    set out20.NBBO;
    if Spread gt 5 and Best_Bid lt lm25 then do;
        Best_Bid=.;
        Best_BidSizeShares=.;
    end;
    if Spread gt 5 and Best_Ask gt lp25 then do;
        Best_Ask=.;
        Best_AskSizeShares=.;
    end;
	keep date time_m sym_root Best_Bidex Best_Bid Best_BidSizeShares Best_Askex 
         Best_Ask Best_AskSizeShares Qu_SeqNum;
run;

/* STEP 4: OUTPUT NEW NBBO RECORDS - IDENTIFY CHANGES IN NBBO RECORDS 
   (CHANGES IN PRICE AND/OR DEPTH) */

data out20.NBBO;
    set out20.NBBO;
    if sym_root ne lag(sym_root) 
       or date ne lag(date) 
       or Best_Ask ne lag(Best_Ask) 
       or Best_Bid ne lag(Best_Bid) 
       or Best_AskSizeShares ne lag(Best_AskSizeShares) 
       or Best_BidSizeShares ne lag(Best_BidSizeShares); 
run;

/* STEP 5: CLEAN DTAQ QUOTES DATA */

data out20.quoteAB;
    set out20.quoteAB;

    /* Create spread and midpoint*/;
    Spread=Ask-Bid;

	/* Delete if abnormal quote conditions */
    if Qu_Cond not in ('A','B','H','O','R','W')then delete; 

	/* Delete if abnormal crossed markets */
    if Bid>Ask then delete;

	/* Delete abnormal spreads*/
    if Spread>5 then delete;

	/* Delete withdrawn Quotes. This is 
	   when an exchange temporarily has no quote, as indicated by quotes 
	   with price or depth fields containing values less than or equal to 0 
	   or equal to '.'. See discussion in Holden and Jacobsen (2020), 
	   page 11. */
    if Ask le 0 or Ask =. then delete;
    if Asksiz le 0 or Asksiz =. then delete;
    if Bid le 0 or Bid =. then delete;
    if Bidsiz le 0 or Bidsiz =. then delete;
	drop  Bidex Askex Qu_Cancel RPI SSR LULD_BBO_CQS 
         LULD_BBO_UTP FINRA_ADF_MPID SIP_Message_ID Part_Time RRN TRF_Time 
         Spread NATL_BBO_LULD;
run;

/* STEP 6: CLEAN DAILY TRADES DATA - DELETE ABNORMAL TRADES */

data out20.trade;
    set out20.trade;
    where Tr_Corr eq '00' and price gt 0;
	drop Tr_Corr Tr_Source TR_RF Part_Time RRN TRF_Time Tr_SCond 
         Tr_StopInd;
run;

/* STEP 7: THE NBBO FILE IS INCOMPLETE BY ITSELF (IF A SINGLE EXCHANGE 
   HAS THE BEST BID AND OFFER, THE QUOTE IS INCLUDED IN THE QUOTES FILE, BUT 
   NOT THE NBBO FILE). TO CREATE THE COMPLETE OFFICIAL NBBO, WE NEED TO 
   MERGE WITH THE QUOTES FILE (SEE FOOTNOTE 6 AND 24 IN OUR PAPER) */


/*
24 In certain instances, when a single exchange has both the best bid and the best offer, then
the official SIP NBBO quote is recorded in the DTAQ Quotes file, not in the DTAQ NBBO file.
When this happens, the field “National BBO Ind” is set equal to 1 (for NYSE, AMEX, and regional
stocks) or else the field “NASDAQ BBO Ind” is set equal to 4 (for NASDAQ stocks). The DTAQ
NBBO file is therefore incomplete because it is missing these records. We construct the Complete
Official NBBO by adding these single-exchange NBBO quotes from the DTAQ Quotes file to the
DTAQ NBBO file. Specifically, we interweave these records by Symbol, Date, Time, and Sequence
Number.

*/

data out20.quoteAB (rename=(Ask=Best_Ask Bid=Best_Bid));
    set out20.quoteAB;
    where (Qu_Source = "C" and NatBBO_Ind='1') or (Qu_Source = "N" and NatBBO_Ind='4');
    keep date time_m sym_root Qu_SeqNum Bid Best_BidSizeShares Ask Best_AskSizeShares;

	/*	Bid/Ask size are in round lots, replace with new shares variable
	and rename Best_BidSizeShares and Best_AskSizeShares*/
	Best_BidSizeShares = Bidsiz * 100;
	Best_AskSizeShares = Asksiz * 100;
run;

proc sort data=out20.NBBO;
    by sym_root date Qu_SeqNum;
run;

proc sort data=out20.quoteAB;
    by sym_root date Qu_SeqNum;
run;

data out20.OfficialCompleteNBBO (drop=Best_Askex Best_Bidex);
    set out20.NBBO out20.quoteAB;
    by sym_root date Qu_SeqNum;
run;

/* If the NBBO Contains two quotes in the exact same microseond, assume 
   last quotes (based on sequence number) is active one */
proc sort data=out20.OfficialCompleteNBBO;
    by sym_root date time_m descending Qu_SeqNum;
run;

proc sort data=out20.OfficialCompleteNBBO nodupkey;
    by sym_root date time_m;
run;

/* STEP 8: INTERLEAVE TRADES WITH NBBO QUOTES. DTAQ TRADES AT NANOSECOND 
   TMMMMMMMMM ARE MATCHED WITH THE DTAQ NBBO QUOTES STILL IN FORCE AT THE 
   NANOSECOND TMMMMMMMM(M-1) */;

data out20.OfficialCompleteNBBO;
    set out20.OfficialCompleteNBBO;type='Q';
    time_m=time_m+.000000001;
	drop Qu_SeqNum;
run;

proc sort data=out20.OfficialCompleteNBBO;
    by sym_root date time_m;
run;



/*==================================================================================================================
                   CHECKING DATA SET IS NON-EMPTY
=====================================================================================================================*/
%NonZero(out20.OfficialCompleteNBBO);
/*=====================================================================================================================*/



%if &N_OBS>0 %then %do;




proc sort data=out20.trade;
    by sym_root date time_m Tr_SeqNum;
run;

data out20.TradesandCorrespondingNBBO;
    set out20.OfficialCompleteNBBO out20.trade;
    by sym_root date time_m type;
	if type='Q' then midpoint=(Best_Ask+Best_Bid)/2;
	else midpoint=.;
run;





data out20.TradesandCorrespondingNBBO (drop=Best_Ask Best_Bid Best_AskSizeShares Best_BidSizeShares Midpoint);
    set out20.TradesandCorrespondingNBBO;
    by sym_root date;
    retain QTime NBO NBB NBOqty NBBqty;
    if first.sym_root or first.date and type='T' then do;
		QTime=.;
        NBO=.;
        NBB=.;
        NBOqty=.;
        NBBqty=.;
    end;
    if type='Q' then Qtime=time_m;
        else Qtime=Qtime;
    if type='Q' then NBO=Best_Ask;
        else NBO=NBO;
    if type='Q' then NBB=Best_Bid;
        else NBB=NBB;
    if type='Q' then NBOqty=Best_AskSizeShares;
        else NBOqty=NBOqty;
    if type='Q' then NBBqty=Best_BidSizeShares;
        else NBBqty=NBBqty;
	format Qtime TIME20.9;
run;

/* STEP 9: CLASSIFY TRADES AS "BUYS" OR "SELLS" USING THREE CONVENTIONS:
   LR = LEE AND READY (1991), EMO = ELLIS, MICHAELY, AND O'HARA (2000)
   AND CLNV = CHAKRABARTY, LI, NGUYEN, AND VAN NESS (2006); DETERMINE NBBO 
   MIDPOINT AND LOCKED AND CROSSED NBBOs */

data out20.TradesandCorrespondingNBBO;
    set out20.TradesandCorrespondingNBBO;
    midpoint=(NBO+NBB)/2;
	/*MODIFICATION FROM HOLDEN AND JACOBSEN MADE HERE, WE ARE KEEPING BOTH TRADES AND QUOTES*/
	IF type='T' THEN DO;
    	if NBO=NBB then lock=1;else lock=0;
    	if NBO<NBB then cross=1;else cross=0;
		END;
run;

/* Determine Whether Trade Price is Higher or Lower than Previous Trade 
   Price, or "Trade Direction" */
data out20.TradesandCorrespondingNBBO;
    set out20.TradesandCorrespondingNBBO;
    by sym_root date;
	retain direction2;
	if type='T' then do;
    	direction=dif(price);
	    if first.sym_root or first.date then direction=.;
	    if direction ne 0 then direction2=direction; 
	    else direction2=direction2;
	end;
	drop direction;
run;

/* First Classification Step: Classify Trades Using Tick Test */
data out20.TradesandCorrespondingNBBO;
    set out20.TradesandCorrespondingNBBO;
    if direction2>0 then BuySellLR=1;
    if direction2<0 then BuySellLR=-1;
    if direction2=. then BuySellLR=.;	
	
    
run;

/* Second Classification Step: Update Trade Classification When 
   Conditions are Met as Specified by LR, EMO, and CLNV */
data out20.TradesandCorrespondingNBBO;
    set out20.TradesandCorrespondingNBBO;
    if lock=0 and cross=0 and price gt midpoint then BuySellLR=1;
    if lock=0 and cross=0 and price lt midpoint then BuySellLR=-1;

	if type='Q' then BuySellLR=0;
    
run;

/* STEP 10: CALCULATE QUOTED SRPEADS AND DEPTHS */

/* Use Quotes During Normal Market Hours */
data out20.TradesandCorrespondingNBBO;
    set out20.TradesandCorrespondingNBBO;
    if time_m < ("9:30:00.000000000"t) or time_m > ("16:00:00.000000000"t)  then delete;
		rename sym_root=ric;
run;









data out20.TradesandCorrespondingNBBO;
set out20.TradesandCorrespondingNBBO;
	by ric date time_m;
*Combining Date and Time into one variable;
/*I want interval to start at 9:30 by default it starts at 12 midnigh*/
/*USING INTNX WE ARE DRAGGING TIME TO BEGINNING OF INTERVAL AND THAN MOVING FORWARD TO END OF INTERVAL SO WE
	OBTAIN THE LAST OBSERVATION IN EACH INTERVAL WHEN WE SORT BY TIME_M BELOW*/
			time_m2=intnx('second1', time_m, 1, 'BEGINNING');
			format time_m2 time20.9;
	date_time=dhms(date, 0,0,time_m2);
	format date_time datetime23.3;

	midpoint= (NBB+NBO)/2;
run;

proc sort data=out20.TradesandCorrespondingNBBO;by ric date time_m;run;
/*Saving firm date and time*/
data _null_;
		set out20.TradesandCorrespondingNBBO (obs=1);
		by ric date time_m;
						open_date=dhms(date, 0,0,time_m2);
						call symput('open_date',open_date);

run;

data out20.TradesandCorrespondingNBBO;
set out20.TradesandCorrespondingNBBO;
	by ric date time_m;
	*open_date is the first date in the entire data set;
	*Interval is the number of 1 second intervals starting from the first peroid;
	interval= intck('second1',&open_date,date_time);

	if BuySellLR=. then BuySellLR=0;


	run;


proc sort data=out20.TradesandCorrespondingNBBO;
	by  ric date interval time_m;
run;




proc sql noprint;
	create table out20.TradesandCorrespondingNBBO as
	select * , 
	sum(BuySellLR) as orderflow, 
	sum(price*size) as volume,
	sum(size*price*BuySellLR) as v_orderflow,
	sum(size*BuySellLR) as s_orderflow,
	sum(type="T") as trades,
	sum(type="Q") as quotes

from out20.TradesandCorrespondingNBBO 
group by interval
order by ric, date, time_m2, time_m;
quit;

data out20.trade; 
set out20.trade; 
rename sym_root=ric;
run;



proc sort data=out20.trade; by ric date time_m;run;
data out20.trade; 
set out20.trade; 
by ric date time_m;
		time_m2=intnx('second1', time_m, 1, 'BEGINNING');
		format time_m2 time20.9;
	date_time=dhms(date, 0,0,time_m2);
	interval= intck('second1',&open_date,date_time);

	format date_time datetime23.3;
run;

data out20.trade; 
set out20.trade; 
by ric date time_m2 time_m;
if last.time_m2 or first.date;
run;


/*sorting by date time_m2*/

/*ORDER FLOW IS ZERO FOR FIRST OBSERVATION. The first observation is only required for first peroid returns
so that all returns correspond to the peroid over which order flow is summed*/
data out20.TradesandCorrespondingNBBO ;
					set out20.TradesandCorrespondingNBBO ;
					by ric date time_m2 time_m;
					
					lagmidpoint=lag(midpoint);
					if first.ric or first.date then lagmidpoint=.;


						if date=lag(date) then do;
						end;

					/*========================*/
						/*KEEP THE FIRST OBSERVATION*/
						if last.time_m2 or first.date;

					

	
run;


proc sql;
	create table out20.TradesandCorrespondingNBBO 
	as select a.*, b.price
	from out20.TradesandCorrespondingNBBO(drop=price) A left join out20.trade B
	on a.ric=b.ric & a.interval = b.interval & a.date=b.date;
quit;



proc sort data=out20.TradesandCorrespondingNBBO;
	by  ric date time_m2 time_m;
run;
data out20.TradesandCorrespondingNBBO;
set out20.TradesandCorrespondingNBBO;
by  ric date time_m2 time_m;
if first.date then do;
s_orderflow=.;
v_orderflow=.;
volume=.;
orderflow=.;
trades=.;
quote=.;
open_price=put("True",$5.);
end;
else do;
open_price=put("False",$5.);
end;
run;



proc append base=taq_20.TAQ_SP_500_2020_1sec data=out20.TradesandCorrespondingNBBO  force;
run;

%end;

%mend SCANLOOP_0;






*EXECUTE MACROS;
/* Macro to SCAN through DATALOG */
%MACRO SCANLOOP_First(SCANFILE,FIELD1,FIELD2, FIELD3, FIELD4);
/* First obtain the number of */
/* records in DATALOG */

DATA _NULL_;
IF 0 THEN SET &SCANFILE NOBS=X;
CALL SYMPUT('RECCOUNT',X);
STOP;
RUN;
/* loop from one to number of */
/* records */
%DO I=1 %TO &RECCOUNT;
/* Advance to the Ith record */
DATA _NULL_;
SET &SCANFILE (FIRSTOBS=&I);
/* store the variables */
/* of interest in */
/* macro variables */
CALL SYMPUT('VAR1',&FIELD1);
CALL SYMPUT('VAR2',&FIELD2);
CALL SYMPUT('VAR3',&FIELD3);
CALL SYMPUT('VAR4',&FIELD4);



STOP;
RUN;



/* now perform the tasks that */
/* wish repeated for each */
/* observation */




%SCANLOOP_0(&VAR1,&VAR2,&VAR3,&VAR4)



%END;
%MEND SCANLOOP_First;




/*=================================================================================================================================*/
/*=================================================================================================================================*/
/*=================================================================================================================================*/
/*=================================================================================================================================*/
/*=================================================================================================================================*/
/*=================================================================================================================================*/
/*=================================================================================================================================*/
/*=================================================================================================================================*/



/*n= number of stocks*/
/*you can add more tickers here*/

data out20.sp_500_2020_500;
do n=1 to 16;
output;
end;
run;

data out20.sp_500_2020_500;
	set out20.sp_500_2020_500;
	if n=1 then symbol_root=PUT("SPY",$8.);
	if n=2 then symbol_root=PUT("AAPL",$8.);
	if n=3 then symbol_root=PUT("MSFT",$8.);
	if n=4 then symbol_root=PUT("TSLA",$8.);
	if n=5 then symbol_root=PUT("AMZN",$8.);
	if n=6 then symbol_root=PUT("ORCL",$8.);
	/*MEME STOCKS - 2020*/
	if n=7 then symbol_root=PUT("GME",$8.);
	if n=8 then symbol_root=PUT("ATOM",$8.);
	if n=9 then symbol_root=PUT("BBBY",$8.);
	if n=10 then symbol_root=PUT("AMC",$8.);
	if n=11 then symbol_root=PUT("HHC",$8.);

	/*FINANCIALS*/
	if n=12 then symbol_root=PUT("AXP",$8.);
	if n=13 then symbol_root=PUT("C",$8.);
	if n=14 then symbol_root=PUT("GS",$8.);
	if n=15 then symbol_root=PUT("JPM",$8.);
	if n=16 then symbol_root=PUT("BAC",$8.);

	




	/*THESE ARE TAQ FILENAMES*/
	nbbo=cats('nbbomsec.nbbom_',put(2020,4.),':');
	ctm=cats('ctmsec.ctm_',put(2020,4.),':');
	cqm=cats('cqmsec.cqm_',put(2020,4.),':');
RUN;

%SCANLOOP_First(out20.sp_500_2020_500,symbol_root,nbbo,cqm,ctm);





/*=================================================*/
/*=================================================*/
/*THIS IS WHERE FILES WILL BE SAVED - CHANGE TO YOUR OWN SCRATCH DIRECTORY WHICH YOU HAVE ACCES TOO*/
proc export data=taq_20.TAQ_SP_500_2020_1sec
/*change your own directory*/
    outfile="/scratch/hecca/saad/taq_20.TAQ_SP_500_2020_1sec.csv"
	/*e.g. libname taq_20  '/scratch/hecca/TradingClub'; */
    dbms=csv
    replace;
run;
/*=================================================*/
/*=================================================*/
/*=================================================*/
/*=================================================*/
