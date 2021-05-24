using System.Collections;
using System.Collections.Generic;
using UnityEngine;
//using UnityEditor;
using System.IO;
using UnityEngine.UI;
using System;

// look into .csv comma seperated

public class ExportData : MonoBehaviour {

	public string filename = "results.csv";	// Export File
	public PLAYERCONTROLLER player;		// Get the PlayerController to access it's variables
	public Timer timer;					// Get the Timer to access it's variables
    public bool cheating = false;		// Is the trial a cheating trial?
	public string game_type = "";		// What is the current game type?
	public bool finish = false;			// Did they finish the game?
	public int forward_count = 0; 		// Total number of forward moves
	public int panic_count = 0;			// Total number of button presses while light is red
	public int trial = 0;				// Current number of trial

	public Manager manager;					// Get the Manager to find the gametype

	public System.Collections.Generic.List<Data> trials = new System.Collections.Generic.List<Data>();				// Array containing all the trials

    public GameObject mob_text;                // Get Mobile text box

    public string txt = "";

    public GameObject pop_up;           // Get the pop up script

    public ErrorPopUp pops;

    public void Get_Filename(string newFilename)
    {
        filename = newFilename;

        Debug.Log("Filename = " + filename);
    }

    void Start()
    {
     //   #if UNITY_EDITOR
    //        Debug.Log("Unity Editor");
    //    #endif

     //   #if UNITY_STANDALONE_WIN
    //        Debug.Log("Stand Alone Windows");
    //    #endif

        #if UNITY_ANDROID
            Debug.Log("Android");
        #endif
    }

    // class for holding the export data for each run
    public class Data
	{
		public bool cheats;				// is the game cheating
		public string mode;				// what is the game mode
		public bool finished;			// did the player finish
		public int steps;				// how many steps were made
		public int hops;				// how many hops were made
		public int set_backs;			// how many times the player was set back
		public int presses;				// how many times did the player press a button
		public int panic;				// how many times the player panic press a button when light is red
		public string time;				// what was the time when the trial finished
		public int trial_count;			// what trial is it
		public int AI_Cheat;			// Probability that an AI will cheat
		public int AI_Level;			// Level of AI

		// empty constructor
		public Data()
		{
			cheats = false;
			mode = "";
			finished = false;
			steps = 0;
			hops = 0;
			set_backs = 0;
			presses = 0;
			panic = 0;
			time = "";
			trial_count = 0;
			//AI_Cheat = 0;
			//AI_Level = 0;
		}

		// constructor
		public Data(bool che, string mod, bool fin, int ste, int hop, int setb, int pre, string tim, int tri)
		{
			cheats = che;
			mode = mod;
			finished = fin;
			steps = ste;
			hops = hop;
			set_backs = setb;
			presses = pre;
			time = tim;
			panic = pre - (ste + hop);
			trial_count = tri;
			//AI_Cheat = aiC;
			//AI_Level = aiL;
		}
	}

	// update the total forward movements and the button presses
	public IEnumerator updateCounts()
	{
		while(true)
		{
			forward_count = player.step_count + player.jump_count;
			panic_count = (player.step_presses + player.jump_presses) - forward_count;
			yield return null;
		}
	}

	public void Start_Recording()
	{
        StopAllCoroutines();
        forward_count = 0;
        panic_count = 0;
        player = GameObject.Find("Player").GetComponent<PLAYERCONTROLLER>();
        timer = GameObject.Find("TimerController").GetComponent<Timer>();
		manager = GameObject.Find("GameManager").GetComponent<Manager>();

		if(manager.trial_list[manager.trial_num - 1].num_players == 2)
		{
			if(manager.trial_list[manager.trial_num - 1].hop_mode)
			{
				game_type = "2 Player Hop";
			}
			else
			{
				game_type = "2 Player Skip";
			}
		}
		else
		{
			if(manager.hop_mode)
			{
				game_type = "1 Player Hop";
			}
			else
			{
				game_type = "1 Player Skip";
			}
		}

		// update trial num
		trial = manager.trial_num;

		StartCoroutine (updateCounts());
	}

	// records the trial data to an array
	public void WriteTrialData()
	{
		string times = timer.timerText.text.ToString();


		Data tmp = new Data (cheating, game_type, finish, player.step_count, player.jump_count, player.setback_count, 
				player.jump_presses + player.step_presses, times, trial);

		trials.Add(tmp);
	}

	// exports the run data to a file
	public void WriteToFile()
	{

        #if UNITY_STANDALONE_WIN
        {
            try{
                var sr = File.CreateText (filename);

                sr.WriteLine ("Number of Players," + "Hop Possible?," + "Trial," + "Steps," + "Hops," + "Set Backs," + "Panic Presses," + "Time," + "Won Trial,"
                                + "AI Level," + "AI Cheat Probability");

                for(int i = 0; i < trials.Count; i++)
                {
                    sr.Write (manager.trial_list[i].num_players + ",");
                    if (manager.trial_list[i].hop_mode)
                        sr.Write("Hop");
                    else
                        sr.Write("Step and Hop");
                    sr.WriteLine("," + trials[i].trial_count + "," + trials[i].steps + "," + trials[i].hops + "," + trials[i].set_backs
                    + "," + trials[i].panic + "," + trials[i].time + "," + manager._TRIAL_WON[i] + "," + manager.trial_list[i].ai_level
                    + "," + manager.trial_list[i].ai_cheat);

                }

                sr.Close ();
            }
            catch (Exception e){
                //Debug.LogException(e, this);
                Debug.Log("Caught Exception");
                pop_up = manager.ErrorPop;
                pops = manager.Pops;
                pop_up.SetActive(true);
                pops.PopUp("Error occured while opening the results file. Please make sure that the file is closed");
            }
        }
        #endif

        #if UNITY_ANDROID
        {
            mob_text = GameObject.Find("Canvas");

            //Text txt = mob_text.GetComponent<Text>();

            //mob_text.SetActive(true);

            txt += "Number of Players\t" + "Hop Possible?\t" + "Trial\t" + "Steps\t" + "Hops\t" + "Set Backs\t" + "Panic Presses\t" + "Time\t" + "Won Trial\t"
                            + "AI Level\t" + "AI Cheat Probability\n";

            for(int i = 0; i < trials.Count; i++)
            {
                txt += manager.trial_list[i].num_players + "\t";
                if (manager.trial_list[i].hop_mode)
                        txt += "Hop\t";
                else
                    txt += "Step and Hop\t";
                txt += ("\t" + trials[i].trial_count + "\t" + trials[i].steps + "\t" + trials[i].hops + "\t" + trials[i].set_backs);
                txt += ("\t" + trials[i].panic + "\t" + trials[i].time + "\t" + manager._TRIAL_WON[i] + "\t" + manager.trial_list[i].ai_level);
                txt += ("\t" + manager.trial_list[i].ai_cheat);
            }
        }
        #endif

		//sr.WriteLine ("\tTrial 1");
		//sr.WriteLine ("Steps: \t" + player.step_count);
		//sr.WriteLine ("Jumps: \t" + player.jump_count);
		//sr.WriteLine ("Set Backs: \t" + player.setback_count);
		//sr.WriteLine ("Time: \t" + timer.minutes.ToString ("00") + ":" + timer.seconds.ToString ("00") + ":" + timer.milliseconds.ToString ("000"));
		
	}

    public void printScreen()
    {
        mob_text = GameObject.FindWithTag("MobileText");

        Text text = mob_text.GetComponent<Text>();

        text.text = txt;




    }

    public void saveFileName()
    {
        GameObject file_name_box = GameObject.FindWithTag("FileNameBox");

        Get_Filename(file_name_box.GetComponent<Text>().text);


    }


}
